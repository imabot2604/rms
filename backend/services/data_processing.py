import pandas as pd
import pandas as pd
import numpy as np
import io
import re
import logging

logger = logging.getLogger(__name__)


def _clean_cell(val):
    """Strip ="" wrappers from export-formatted cells and convert to numeric if possible."""
    if pd.isna(val):
        return val
    s = str(val).strip()
    # Remove ="" wrapper
    if s.startswith('="') and s.endswith('"'):
        s = s[2:-1]
    elif s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    s = s.strip()
    if s == '' or s.lower() == 'nan':
        return np.nan
    # Try numeric conversion (remove commas, handle percentages)
    try:
        clean = s.replace(',', '')
        if clean.endswith('%'):
            return float(clean[:-1])
        return float(clean)
    except ValueError:
        return s


def _find_header_row(df_raw, max_scan=25):
    """Scan the first N rows to find the one containing month headers."""
    month_tokens = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    for idx in range(min(max_scan, len(df_raw))):
        row = df_raw.iloc[idx]
        date_col_count = 0
        for val in row:
            if pd.isna(val):
                continue
            val_str = str(val).lower().strip()
            # Remove ="" wrappers for comparison
            if val_str.startswith('="'):
                val_str = val_str[2:]
            if val_str.endswith('"'):
                val_str = val_str[:-1]
            val_str = val_str.strip()

            # Check if it's a timestamp object
            if isinstance(val, (pd.Timestamp, np.datetime64)):
                date_col_count += 1
                continue

            # Check for month name patterns like "Jan, 2023" or "Jan-23"
            for m in month_tokens:
                if m in val_str:
                    # Exclude range strings like "Jan, 2023 - Dec, 2023"
                    if ' - ' in val_str or ' to ' in val_str:
                        continue
                    date_col_count += 1
                    break

        if date_col_count >= 3:
            return idx
    return None


def _extract_month_columns(columns):
    """Identify which columns represent individual months (not ranges/totals)."""
    month_tokens = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    month_cols = []
    for col in columns:
        col_str = str(col).lower().strip()
        if isinstance(col, (pd.Timestamp, np.datetime64)):
            month_cols.append(col)
            continue
        # Skip range columns like "Jan, 2023 - Dec, 2023" or "Total"
        if ' - ' in col_str or ' to ' in col_str or 'total' in col_str:
            continue
        for m in month_tokens:
            if m in col_str:
                month_cols.append(col)
                break
    return month_cols


# Canonical schema mapping rules shared by header-based and USALI stat-block
# (row-label-based) ingestion. Priority-ordered: first match wins per target.
# NOTE: more specific rules MUST come before generic ones (e.g. 'total
# departmental income' before 'total operating revenue').
_SCHEMA_RULES = [
    ('Date', lambda s: s == 'date' or 'date' in s or 'month' in s or 'period' in s),
    ('Rooms_Available', lambda s: ('avail' in s and 'room' in s) or s == 'rooms available'
                       or 'available room' in s or 'room nights available' in s),
    ('Rooms_Sold', lambda s: (('sold' in s or 'demand' in s) and 'room' in s)
                   or s == 'rooms sold' or 'room nights sold' in s),
    ('Occupancy_Pct', lambda s: ('occ' in s and '%' in s) or s == 'occupancy %'
                     or s == 'occupancy' or s == 'occupancy percent' or s.startswith('occupancy')),
    ('ADR', lambda s: s == 'adr' or ('avg' in s and 'rate' in s)
            or 'average daily rate' in s),
    ('RevPAR', lambda s: 'revpar' in s or 'revenue per available room' in s),
    ('Room_Revenue', lambda s: ('room' in s and ('rev' in s or 'department' in s) and 'total' in s)
                     or s == 'room department' or s == 'rooms revenue'
                     or s == 'room revenue' or s == 'total rooms revenue'),
    ('FB_Revenue', lambda s: 'f&b' in s or ('food' in s and 'bev' in s)
                   or s == 'food and beverages department' or s == 'f&b revenue'),
    # Other operated departments / misc income lines (needed so revenue
    # reconciles: Room + F&B + Other == Total Revenue).
    ('Other_Revenue', lambda s: 'other operated' in s or 'minor operated' in s
                      or s == 'other revenue' or 'miscellaneous income' in s
                      or 'rentals and other income' in s or s == 'other income'),
    ('Total_Departmental_Income', lambda s: 'total departmental income' in s
                                  or 'total department income' in s
                                  or 'departmental profit' in s),
    ('Total_UOE', lambda s: 'total undistributed' in s or 'uoe' in s
                  or 'total undistributed operating expenses' in s),
    ('GOP', lambda s: 'gross operating profit' in s or s == 'gop' or s == 'gop %'),
    # NOI: accept USALI 'Net Income (loss)' / 'Net Income loss' variants in
    # addition to 'Net Operating Income'. Normalized to canonical NOI.
    ('NOI', lambda s: 'net operating income' in s or s == 'noi'
            or 'net income (loss)' in s or 'net income loss' in s
            or s == 'net income' or 'ebitda' == s),
    ('Total_Revenue', lambda s: 'total operating revenue' in s
                      or s == 'total revenue' or 'total hotel revenue' in s),
]


def _match_schema_target(label, mapped_targets):
    """Return the canonical target for a single label, or None."""
    s = str(label).lower().strip()
    for target, matcher in _SCHEMA_RULES:
        if target in mapped_targets:
            continue
        try:
            if matcher(s):
                return target
        except Exception:
            continue
    return None


def _map_columns_to_schema(columns):
    """Map column names to our canonical schema using heuristics."""
    col_map = {}
    mapped_targets = set()
    for col in columns:
        target = _match_schema_target(col, mapped_targets)
        if target is not None:
            col_map[col] = target
            mapped_targets.add(target)
    return col_map


def process_uploaded_files(files):
    """
    Reads multiple Excel/CSV files from memory, standardizes to a canonical
    hospitality schema, and concatenates them into a single DataFrame.
    Handles wide-format P&L exports (months as columns) with ="" escaping.
    """
    dfs = []

    for file in files:
        contents = file.file.read()
        filename = getattr(file, 'filename', '') or ''

        try:
            if filename.lower().endswith('.csv') or filename.lower().endswith('.txt'):
                # Try comma first
                try:
                    df_raw = pd.read_csv(io.BytesIO(contents), header=None, dtype=str)
                    if df_raw.shape[1] <= 1:
                        df_tab = pd.read_csv(io.BytesIO(contents), header=None, dtype=str, sep='\t')
                        if df_tab.shape[1] > df_raw.shape[1]:
                            df_raw = df_tab
                except Exception:
                    df_raw = pd.read_csv(io.BytesIO(contents), header=None, dtype=str, sep='\t')
            else:
                df_raw = pd.read_excel(io.BytesIO(contents), header=None, dtype=str)
        except Exception as e:
            logger.warning(f"Primary read failed for {filename}, trying CSV/TXT fallback: {e}")
            try:
                df_raw = pd.read_csv(io.BytesIO(contents), header=None, dtype=str)
                if df_raw.shape[1] <= 1:
                    df_tab = pd.read_csv(io.BytesIO(contents), header=None, dtype=str, sep='\t')
                    if df_tab.shape[1] > df_raw.shape[1]:
                        df_raw = df_tab
            except Exception:
                raise ValueError(f"Could not read file: {filename}")

        if df_raw.empty:
            continue

        # 1. Find the header row
        header_row_idx = _find_header_row(df_raw)

        if header_row_idx is not None:
            header_values = df_raw.iloc[header_row_idx].apply(
                lambda v: _clean_cell(v) if not pd.isna(v) else v
            )
            df = df_raw.iloc[header_row_idx + 1:].reset_index(drop=True)
            df.columns = header_values
        else:
            # Fallback: use first row as header
            header_values = df_raw.iloc[0].apply(
                lambda v: _clean_cell(v) if not pd.isna(v) else v
            )
            df = df_raw.iloc[1:].reset_index(drop=True)
            df.columns = header_values

        # 2. Clean column names
        clean_cols = []
        for i, c in enumerate(df.columns):
            if pd.isna(c) or str(c).strip() == '' or str(c).strip().lower() == 'nan':
                clean_cols.append(f"_unnamed_{i}")
            else:
                cleaned = str(c).strip()
                # Remove residual ="" wrappers
                if cleaned.startswith('="') and cleaned.endswith('"'):
                    cleaned = cleaned[2:-1].strip()
                clean_cols.append(cleaned)
        df.columns = clean_cols

        # 3. Clean all cell values
        for col in df.columns:
            df[col] = df[col].apply(_clean_cell)

        # 4. Drop rows that are entirely NaN or have an empty first column (section headers)
        first_col = df.columns[0]
        df = df.dropna(how='all')
        df = df[df[first_col].notna()].reset_index(drop=True)

        # 5. Detect wide format (months as columns)
        month_cols = _extract_month_columns(df.columns)
        is_wide = len(month_cols) >= 3

        if is_wide:
            id_col = df.columns[0]

            # Filter out rows where the id_col is NaN or empty
            df = df[df[id_col].notna() & (df[id_col] != '')].reset_index(drop=True)

            # Convert id_col to string
            df[id_col] = df[id_col].astype(str).str.strip()

            # Melt from wide to long
            df_long = df.melt(id_vars=[id_col], value_vars=month_cols,
                              var_name='Date', value_name='Value')

            # Remove rows where Value is NaN
            df_long = df_long.dropna(subset=['Value'])

            # Deduplicate: keep the FIRST occurrence of each (Date, KPI) pair
            # In hospitality P&Ls, duplicate row names appear across Revenue/Expense/Income sections
            df_long = df_long.drop_duplicates(subset=['Date', id_col], keep='first')

            # Pivot to get KPIs as columns
            try:
                df = df_long.pivot(index='Date', columns=id_col, values='Value').reset_index()
                df.columns.name = None
            except Exception as e:
                raise ValueError(f"Failed to reshape data: {e}")

        # 6. Map columns to canonical schema
        col_map = _map_columns_to_schema(df.columns)
        df = df.rename(columns=col_map)

        # 7. Ensure we have critical columns; try to compute missing ones
        if 'Date' in df.columns:
            df['Date'] = _parse_month_series(df['Date'])
            df = df.dropna(subset=['Date'])

        if 'Occupancy_Pct' in df.columns:
            df['Occupancy_Pct'] = pd.to_numeric(df['Occupancy_Pct'], errors='coerce')
            if df['Occupancy_Pct'].max(skipna=True) > 1:
                df['Occupancy_Pct'] = df['Occupancy_Pct'] / 100.0

        for numeric_col in ['ADR', 'RevPAR', 'Room_Revenue', 'Rooms_Available',
                            'Rooms_Sold', 'FB_Revenue', 'GOP', 'NOI', 'Total_Revenue', 'Total_UOE']:
            if numeric_col in df.columns:
                df[numeric_col] = pd.to_numeric(df[numeric_col], errors='coerce')

        # Compute RevPAR if missing but ADR and Occupancy exist
        if 'RevPAR' not in df.columns and 'ADR' in df.columns and 'Occupancy_Pct' in df.columns:
            df['RevPAR'] = df['ADR'] * df['Occupancy_Pct']

        # Compute Rooms_Sold if missing but Occupancy and Rooms_Available exist
        if 'Rooms_Sold' not in df.columns and 'Occupancy_Pct' in df.columns and 'Rooms_Available' in df.columns:
            df['Rooms_Sold'] = (df['Occupancy_Pct'] * df['Rooms_Available']).round(0)

        # Derive accounting identities (Total_UOE, Other_Revenue) so the COA
        # hierarchy reconciles. Built via a single batched concat (no per-column
        # df.insert fragmentation).
        df = derive_accounting_identities(df)

        dfs.append(df)

    if not dfs:
        raise ValueError("No valid data frames could be extracted from the uploaded files.")

    master_df = pd.concat(dfs, ignore_index=True)

    if 'Date' not in master_df.columns:
        raise ValueError("Could not detect a Date/Month column in the uploaded files. "
                         "Ensure your file has month headers like 'Jan, 2023' or a 'Date' column.")

    # Ensure deterministic datetime dtype before sorting.
    if not np.issubdtype(master_df['Date'].dtype, np.datetime64):
        master_df['Date'] = _parse_month_series(master_df['Date'])
        master_df = master_df.dropna(subset=['Date'])

    master_df = master_df.sort_values(by='Date').reset_index(drop=True)

    # Drop duplicate dates (keep first occurrence, which is typically the most complete)
    master_df = master_df.drop_duplicates(subset=['Date'], keep='first').reset_index(drop=True)

    # Feature Engineering
    master_df = engineer_features(master_df)

    return master_df


def _parse_month_series(series):
    """
    Parse month labels deterministically.

    Fairfield/USALI wide files use labels like 'Jan, 2024'. We try a small set
    of explicit formats first (fast, warning-free, deterministic) and only fall
    back to generic inference for anything unmatched.
    """
    s = series.astype(str).str.strip()
    result = pd.Series(pd.NaT, index=series.index, dtype='datetime64[ns]')

    explicit_formats = ['%b, %Y', '%b %Y', '%B, %Y', '%B %Y', '%b-%y', '%b-%Y', '%Y-%m', '%m/%Y']
    for fmt in explicit_formats:
        mask = result.isna()
        if not mask.any():
            break
        parsed = pd.to_datetime(s[mask], format=fmt, errors='coerce')
        result[mask] = parsed

    # Single generic fallback for any remaining unparsed labels.
    mask = result.isna()
    if mask.any():
        result[mask] = pd.to_datetime(s[mask], errors='coerce')

    return result


def derive_accounting_identities(df):
    """
    Derive Total_UOE and Other_Revenue from available accounting identities so
    the COA hierarchy reconciles even when the source omits these lines.

    Identities used (best-effort, only when inputs exist):
      * Other_Revenue = Total_Revenue - (Room_Revenue + FB_Revenue)
      * Total_UOE     = Total_Departmental_Income - GOP
                        (fallback) Total_Revenue - GOP

    New columns are constructed in a single batched concat to avoid pandas
    DataFrame fragmentation.
    """
    new_cols = {}

    has = lambda c: c in df.columns and not df[c].isna().all()

    # Other_Revenue so Room + F&B + Other == Total_Revenue.
    if 'Other_Revenue' not in df.columns and has('Total_Revenue'):
        room = df['Room_Revenue'] if 'Room_Revenue' in df.columns else 0
        fb = df['FB_Revenue'] if 'FB_Revenue' in df.columns else 0
        other = pd.to_numeric(df['Total_Revenue'], errors='coerce') \
            - pd.to_numeric(room, errors='coerce').fillna(0) \
            - pd.to_numeric(fb, errors='coerce').fillna(0)
        # Only keep meaningful (non-trivial) residuals.
        if other.abs().sum(skipna=True) > 1e-6:
            new_cols['Other_Revenue'] = other

    # Total_UOE fallback derivation so GOP can reconcile.
    if 'Total_UOE' not in df.columns:
        if has('Total_Departmental_Income') and has('GOP'):
            new_cols['Total_UOE'] = pd.to_numeric(df['Total_Departmental_Income'], errors='coerce') \
                - pd.to_numeric(df['GOP'], errors='coerce')
        elif has('Total_Revenue') and has('GOP'):
            new_cols['Total_UOE'] = pd.to_numeric(df['Total_Revenue'], errors='coerce') \
                - pd.to_numeric(df['GOP'], errors='coerce')

    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


def engineer_features(df):
    """Engineers features required by the ML models."""
    if 'Date' not in df.columns:
        return df

    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Trend'] = np.arange(len(df))
    df['Seasonality'] = np.sin(2 * np.pi * df['Month'] / 12)

    if 'ADR' in df.columns:
        df['ADR_Lag'] = df['ADR'].shift(1).bfill()

    # Simulate Comp_Index anchored to actual occupancy
    if 'Occupancy_Pct' in df.columns:
        np.random.seed(42)  # Reproducible results
        comp_occ = np.random.uniform(0.45, 0.75, len(df))
        safe_occ = df['Occupancy_Pct'].replace(0, 0.01).fillna(0.5)
        df['Comp_Index'] = comp_occ / safe_occ

    return df


def generate_otb_pace(df):
    """Constructs a synthetic OTB pace layer based on actual rooms sold."""
    np.random.seed(42)

    if 'Rooms_Sold' not in df.columns or df['Rooms_Sold'].isna().all():
        df['OTB_Pace_Index'] = 1.0
        return df

    pace_data = []
    for _, row in df.iterrows():
        total_sold = row.get('Rooms_Sold')
        if pd.isna(total_sold) or total_sold == 0:
            pace_data.append(1.0)
        else:
            noise = np.random.uniform(0.85, 1.15)
            pace_data.append(noise)

    df['OTB_Pace_Index'] = pace_data
    return df
