import React from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

const KpiCard = ({ title, value, prefix = '', suffix = '', trend = null, trendLabel = 'vs last period' }) => {
  const isPositive = trend > 0;
  const isNegative = trend < 0;
  
  return (
    <div className="card">
      <h3 className="text-secondary text-sm font-medium mb-2">{title}</h3>
      <div className="flex items-baseline gap-2">
        <span className="text-2xl font-bold text-primary">
          {prefix}{value}{suffix}
        </span>
      </div>
      
      {trend !== null && (
        <div className={`flex items-center gap-1 mt-2 text-xs font-medium ${isPositive ? 'text-success' : isNegative ? 'text-danger' : 'text-tertiary'}`}>
          {isPositive ? <TrendingUp size={14} /> : isNegative ? <TrendingDown size={14} /> : <Minus size={14} />}
          <span>{Math.abs(trend).toFixed(1)}%</span>
          <span className="text-tertiary font-normal ml-1">{trendLabel}</span>
        </div>
      )}
    </div>
  );
};

export default KpiCard;
