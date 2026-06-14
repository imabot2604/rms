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


def _map_columns_to_schema(columns):
    """Map column names to our canonical schema using heuristics."""
    col_map = {}
    mapped_targets = set()

    # Priority-ordered mapping rules (first match wins for each target)
    rules = [
        ('Date', lambda s: s == 'date' or 'date' in s or 'month' in s or 'period' in s),
        ('Rooms_Available', lambda s: 'avail' in s and 'room' in s),
        ('Rooms_Sold', lambda s: ('sold' in s or 'demand' in s) and 'room' in s),
        ('Occupancy_Pct', lambda s: 'occ' in s and '%' in s or s == 'occupancy %' or s == 'occupancy'),
        ('ADR', lambda s: s == 'adr' or ('avg' in s and 'rate' in s)),
        ('RevPAR', lambda s: 'revpar' in s),
        ('Room_Revenue', lambda s: ('room' in s and ('rev' in s or 'department' in s) and 'total' in s)
                         or s == 'room department'),
        ('FB_Revenue', lambda s: 'f&b' in s or ('food' in s and 'bev' in s)
                       or s == 'food and beverages department'),
        ('Total_UOE', lambda s: 'total undistributed' in s or 'uoe' in s),
        ('GOP', lambda s: 'gross operating profit' in s or s == 'gop' or s == 'gop %'),
        ('NOI', lambda s: 'net operating income' in s or s == 'noi'),
        ('Total_Revenue', lambda s: 'total operating revenue' in s),
    ]

    for col in columns:
        col_str = str(col).lower().strip()
        for target, matcher in rules:
            if target not in mapped_targets and matcher(col_str):
                col_map[col] = target
                mapped_targets.add(target)
                break

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
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
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

        dfs.append(df)

    if not dfs:
        raise ValueError("No valid data frames could be extracted from the uploaded files.")

    master_df = pd.concat(dfs, ignore_index=True)

    if 'Date' not in master_df.columns:
        raise ValueError("Could not detect a Date/Month column in the uploaded files. "
                         "Ensure your file has month headers like 'Jan, 2023' or a 'Date' column.")

    master_df = master_df.sort_values(by='Date').reset_index(drop=True)

    # Drop duplicate dates (keep first occurrence, which is typically the most complete)
    master_df = master_df.drop_duplicates(subset=['Date'], keep='first').reset_index(drop=True)

    # Feature Engineering
    master_df = engineer_features(master_df)

    return master_df


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
