import * as XLSX from 'xlsx';
import Papa from 'papaparse';

/**
 * Parses an uploaded file (CSV or Excel) and attempts to extract
 * a long-format array of hospitality metrics.
 */
export const parseUploadFile = async (file) => {
  return new Promise((resolve, reject) => {
    const extension = file.name.split('.').pop().toLowerCase();
    
    if (extension === 'csv') {
      Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        complete: (results) => {
          resolve(processParsedData(results.data, file.name));
        },
        error: (error) => {
          reject(error);
        }
      });
    } else if (['xlsx', 'xls'].includes(extension)) {
      const reader = new FileReader();
      reader.onload = (e) => {
        try {
          const data = new Uint8Array(e.target.result);
          const workbook = XLSX.read(data, { type: 'array' });
          
          // For now, take the first sheet
          const firstSheetName = workbook.SheetNames[0];
          const worksheet = workbook.Sheets[firstSheetName];
          
          // Convert sheet to array of objects
          const jsonData = XLSX.utils.sheet_to_json(worksheet, { defval: null });
          resolve(processParsedData(jsonData, file.name));
        } catch (error) {
          reject(error);
        }
      };
      reader.onerror = (error) => reject(error);
      reader.readAsArrayBuffer(file);
    } else {
      reject(new Error('Unsupported file type. Please upload CSV or Excel files.'));
    }
  });
};

/**
 * Normalizes flat array of objects (like a wide CSV) into a standard schema.
 * Handles detection of metrics like "Rooms Available", "ADR", "Occupancy".
 */
const processParsedData = (data, sourceFileName) => {
  // 1. Identify columns
  if (!data || data.length === 0) return { raw: data, normalized: [], quality: [] };
  
  const headers = Object.keys(data[0]);
  
  // Very simplistic mapping logic for prototype
  // In a real app, this would use fuzzy matching and allow user overrides
  const columnMap = {
    date: headers.find(h => /date|month|year|period/i.test(h)),
    roomsAvailable: headers.find(h => /room.*avail/i.test(h) || /inventory/i.test(h)),
    roomsSold: headers.find(h => /room.*sold/i.test(h) || /demand/i.test(h)),
    occ: headers.find(h => /occ/i.test(h)),
    adr: headers.find(h => /adr|rate/i.test(h)),
    revpar: headers.find(h => /revpar/i.test(h)),
    roomRev: headers.find(h => /room.*rev/i.test(h)),
    gop: headers.find(h => /gop|gross.*profit/i.test(h))
  };

  const normalizedData = data.map((row, index) => {
    let dateStr = row[columnMap.date];
    // Attempt basic parsing
    let parsedDate = dateStr ? new Date(dateStr) : new Date();
    
    // Attempt to calculate missing values if possible
    let roomsAvail = parseFloat(row[columnMap.roomsAvailable]) || null;
    let roomsSold = parseFloat(row[columnMap.roomsSold]) || null;
    let occ = parseFloat(row[columnMap.occ]) || null;
    let adr = parseFloat(row[columnMap.adr]) || null;
    let revpar = parseFloat(row[columnMap.revpar]) || null;
    let roomRev = parseFloat(row[columnMap.roomRev]) || null;

    // Derived Logic
    if (roomsSold && roomsAvail && !occ) occ = roomsSold / roomsAvail;
    if (occ && adr && !revpar) revpar = occ * adr;
    if (roomsSold && adr && !roomRev) roomRev = roomsSold * adr;
    
    // Fix percentages > 1 (if occupancy is 75 instead of 0.75)
    if (occ > 1 && occ <= 100) occ = occ / 100;

    return {
      id: `row-${index}`,
      source: sourceFileName,
      date: parsedDate.toISOString(),
      rawDateStr: dateStr,
      metrics: {
        roomsAvailable: roomsAvail,
        roomsSold: roomsSold,
        occupancy: occ,
        adr: adr,
        revpar: revpar,
        roomRevenue: roomRev,
        gop: parseFloat(row[columnMap.gop]) || null,
      },
      raw: row
    };
  }).filter(row => row.metrics.adr !== null || row.metrics.occupancy !== null); // Filter out garbage rows

  // Quality Checks
  const qualityIssues = [];
  if (!columnMap.date) qualityIssues.push({ type: 'warning', msg: 'Could not confidently identify a Date column.' });
  if (!columnMap.adr) qualityIssues.push({ type: 'warning', msg: 'Could not find ADR column.' });
  
  const negativeAdrRows = normalizedData.filter(r => r.metrics.adr < 0);
  if (negativeAdrRows.length > 0) qualityIssues.push({ type: 'error', msg: `Found ${negativeAdrRows.length} rows with negative ADR.` });

  return {
    raw: data,
    normalized: normalizedData,
    quality: qualityIssues,
    columnMap
  };
};
