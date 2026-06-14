import React, { useState, useCallback } from 'react';
import { Upload, AlertCircle, CheckCircle2, ChevronRight, BarChart2, FileSpreadsheet, Loader2 } from 'lucide-react';
import useStore from '../store/useStore';

const DataIngestion = () => {
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [uploadedFileName, setUploadedFileName] = useState(null);

  const {
    setRawData,
    setNormalizedData,
    setActiveModule,
    isDataLoaded,
    normalizedData
  } = useStore();

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragging(true);
    } else if (e.type === 'dragleave') {
      setIsDragging(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    setError(null);

    const files = e.dataTransfer?.files || e.target?.files;
    if (files && files.length > 0) {
      processFiles(files);
    }
  }, []);

  const processFiles = async (files) => {
    setIsProcessing(true);
    setError(null);
    setUploadedFileName(files[0]?.name || 'Unknown file');

    try {
      const formData = new FormData();
      for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
      }

      const response = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let errorMsg = 'Failed to process files on server.';
        try {
          const errorData = await response.json();
          errorMsg = errorData.detail || errorMsg;
        } catch {
          // response wasn't JSON
        }
        throw new Error(errorMsg);
      }

      const result = await response.json();

      if (!result.data || result.data.length === 0) {
        throw new Error('No valid hospitality data could be extracted from these files. Ensure your file contains Occupancy, ADR, or RevPAR data.');
      }

      setRawData(result.data);
      setNormalizedData(result.data);

    } catch (err) {
      console.error('Upload error:', err);
      if (err.message === 'Failed to fetch') {
        setError('Could not connect to the backend server. Make sure the Python backend is running on port 8000.');
      } else {
        setError(err.message || 'Failed to process file.');
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const safeNumber = (val) => {
    if (val === null || val === undefined || isNaN(val)) return null;
    return val;
  };

  return (
    <div className="flex-col gap-6" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div className="card">
        <h2 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '0.5rem' }}>Upload Data</h2>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
          Upload your hotel's monthly income statement, daily operating report, or OTB pace file.
          The system will automatically extract and normalize the KPIs.
        </p>

        <div
          style={{
            border: '2px dashed',
            borderColor: isDragging ? 'var(--accent-primary)' : 'var(--border-medium)',
            borderRadius: '0.75rem',
            padding: '3rem',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            transition: 'all 0.2s ease',
            backgroundColor: isDragging ? 'rgba(59, 130, 246, 0.1)' : 'rgba(0, 0, 0, 0.2)'
          }}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <div style={{
            backgroundColor: 'var(--bg-surface)',
            padding: '1rem',
            borderRadius: '50%',
            marginBottom: '1rem',
            boxShadow: 'var(--shadow-lg)'
          }}>
            <Upload size={32} style={{ color: 'var(--accent-primary)' }} />
          </div>

          <h3 style={{ fontSize: '1.25rem', fontWeight: 600, marginBottom: '0.5rem' }}>Drag & Drop Files Here</h3>
          <p style={{ color: 'var(--text-tertiary)', marginBottom: '1.5rem' }}>Supports .xlsx, .xls, .csv, and .txt</p>

          <input
            type="file"
            id="file-upload"
            style={{ display: 'none' }}
            accept=".csv, .txt, application/vnd.openxmlformats-officedocument.spreadsheetml.sheet, application/vnd.ms-excel"
            onChange={handleDrop}
            multiple
          />
          <label htmlFor="file-upload" className="btn btn-primary" style={{ cursor: 'pointer' }}>
            Browse Files
          </label>
        </div>

        {isProcessing && (
          <div style={{
            marginTop: '1.5rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '0.75rem',
            color: 'var(--accent-primary)'
          }}>
            <Loader2 size={18} style={{ animation: 'spin 1s linear infinite' }} />
            <span>Processing {uploadedFileName}...</span>
          </div>
        )}

        {error && (
          <div style={{
            marginTop: '1.5rem',
            padding: '1rem',
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            borderColor: 'rgba(239, 68, 68, 0.2)',
            border: '1px solid rgba(239, 68, 68, 0.2)',
            borderRadius: '0.5rem',
            display: 'flex',
            gap: '0.75rem',
            color: 'var(--accent-danger)'
          }}>
            <AlertCircle />
            <div>
              <h4 style={{ fontWeight: 600 }}>Upload Error</h4>
              <p style={{ fontSize: '0.875rem' }}>{error}</p>
            </div>
          </div>
        )}
      </div>

      {isDataLoaded && !error && normalizedData.length > 0 && (
        <div className="card" style={{
          borderColor: 'rgba(16, 185, 129, 0.3)',
          backgroundColor: 'rgba(16, 185, 129, 0.05)',
          animation: 'fadeIn 0.4s ease forwards'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div style={{ display: 'flex', gap: '1rem' }}>
              <CheckCircle2 style={{ color: 'var(--accent-success)', marginTop: '0.25rem' }} size={24} />
              <div>
                <h3 style={{ fontSize: '1.125rem', fontWeight: 700, color: 'var(--accent-success)', marginBottom: '0.25rem' }}>
                  Data Successfully Normalized
                </h3>
                <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: '1rem' }}>
                  Extracted {normalizedData.length} time-series records. Ready for simulation.
                </p>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  {normalizedData[0]?.date && <span style={{ padding: '0.25rem 0.5rem', backgroundColor: 'rgba(0,0,0,0.3)', borderRadius: '0.25rem', fontSize: '0.75rem' }}>Dates found</span>}
                  {safeNumber(normalizedData[0]?.metrics?.adr) !== null && <span style={{ padding: '0.25rem 0.5rem', backgroundColor: 'rgba(0,0,0,0.3)', borderRadius: '0.25rem', fontSize: '0.75rem' }}>ADR detected</span>}
                  {safeNumber(normalizedData[0]?.metrics?.occupancy) !== null && <span style={{ padding: '0.25rem 0.5rem', backgroundColor: 'rgba(0,0,0,0.3)', borderRadius: '0.25rem', fontSize: '0.75rem' }}>Occupancy detected</span>}
                  {safeNumber(normalizedData[0]?.metrics?.gop) !== null && <span style={{ padding: '0.25rem 0.5rem', backgroundColor: 'rgba(0,0,0,0.3)', borderRadius: '0.25rem', fontSize: '0.75rem' }}>GOP detected</span>}
                </div>
              </div>
            </div>

            <button
              className="btn btn-primary"
              onClick={() => setActiveModule('dashboard')}
              style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
            >
              <BarChart2 size={16} />
              Go to Dashboard
              <ChevronRight size={16} />
            </button>
          </div>

          <div style={{
            marginTop: '1.5rem',
            backgroundColor: 'rgba(0,0,0,0.2)',
            borderRadius: '0.5rem',
            padding: '1rem',
            border: '1px solid var(--border-light)',
            overflowX: 'auto'
          }}>
            <h4 style={{ fontSize: '0.875rem', fontWeight: 700, marginBottom: '0.75rem', color: 'var(--text-secondary)' }}>Normalized Dataset Preview</h4>
            <table style={{ width: '100%', textAlign: 'left', fontSize: '0.875rem', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ color: 'var(--text-tertiary)', borderBottom: '1px solid var(--border-light)' }}>
                  <th style={{ paddingBottom: '0.5rem', fontWeight: 500 }}>Date</th>
                  <th style={{ paddingBottom: '0.5rem', fontWeight: 500 }}>Occ %</th>
                  <th style={{ paddingBottom: '0.5rem', fontWeight: 500 }}>ADR</th>
                  <th style={{ paddingBottom: '0.5rem', fontWeight: 500 }}>RevPAR</th>
                  <th style={{ paddingBottom: '0.5rem', fontWeight: 500 }}>Room Rev</th>
                  <th style={{ paddingBottom: '0.5rem', fontWeight: 500 }}>GOP</th>
                </tr>
              </thead>
              <tbody>
                {normalizedData.slice(0, 5).map((row, idx) => (
                  <tr key={idx} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', color: 'var(--text-secondary)' }}>
                    <td style={{ padding: '0.5rem 0' }}>{row.date ? new Date(row.date).toLocaleDateString() : '-'}</td>
                    <td style={{ padding: '0.5rem 0' }}>{safeNumber(row.metrics?.occupancy) !== null ? (row.metrics.occupancy * 100).toFixed(1) + '%' : '-'}</td>
                    <td style={{ padding: '0.5rem 0' }}>{safeNumber(row.metrics?.adr) !== null ? '$' + row.metrics.adr.toFixed(2) : '-'}</td>
                    <td style={{ padding: '0.5rem 0' }}>{safeNumber(row.metrics?.revpar) !== null ? '$' + row.metrics.revpar.toFixed(2) : '-'}</td>
                    <td style={{ padding: '0.5rem 0' }}>{safeNumber(row.metrics?.roomRevenue) !== null ? '$' + row.metrics.roomRevenue.toLocaleString() : '-'}</td>
                    <td style={{ padding: '0.5rem 0' }}>{safeNumber(row.metrics?.gop) !== null ? '$' + row.metrics.gop.toLocaleString() : '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {normalizedData.length > 5 && (
              <p style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', marginTop: '0.75rem', textAlign: 'center' }}>
                Showing 5 of {normalizedData.length} records
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default DataIngestion;
