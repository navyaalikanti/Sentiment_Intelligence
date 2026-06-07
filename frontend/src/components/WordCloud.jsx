import React, { useMemo } from 'react';

function stableHash(value) {
  let hash = 0;
  for (let i = 0; i < value.length; i += 1) {
    hash = (hash * 31 + value.charCodeAt(i)) >>> 0;
  }
  return hash;
}

export default function WordCloud({ words = [], emptyLabel = 'No words available.' }) {
  const items = useMemo(() => {
    if (!words.length) return [];
    const max = Math.max(...words.map((item) => item.count || 0), 1);
    return words.map((item) => {
      const ratio = (item.count || 0) / max;
      const fontSize = 14 + ratio * 18;
      const opacity = 0.55 + ratio * 0.45;
      const width = stableHash(item.word) % 6;
      return {
        ...item,
        fontSize,
        opacity,
        width,
      };
    });
  }, [words]);

  if (!items.length) {
    return <div className="text-sm text-slate-400">{emptyLabel}</div>;
  }

  return (
    <div className="flex flex-wrap items-end gap-x-3 gap-y-4 leading-none">
      {items.map((item) => (
        <span
          key={item.word}
          className="inline-flex rounded-full border border-blue-500/10 bg-white/[0.03] px-3 py-1 text-white"
          style={{
            fontSize: `${item.fontSize}px`,
            opacity: item.opacity,
            transform: `translateY(${item.width}px)`,
          }}
          title={`${item.word}: ${item.count}`}
        >
          {item.word}
        </span>
      ))}
    </div>
  );
}
