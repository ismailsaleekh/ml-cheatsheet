/**
 * SearchHighlight - Highlights matching text in search results
 */
import { useMemo } from 'react';
import { cn } from '@/utils/cn';

interface SearchHighlightProps {
  text: string;
  query: string;
  className?: string;
  highlightClassName?: string;
}

/**
 * Component that highlights matching portions of text
 */
export const SearchHighlight: React.FC<SearchHighlightProps> = ({
  text,
  query,
  className,
  highlightClassName,
}) => {
  const parts = useMemo(() => {
    if (!query || !text) {
      return [{ text, highlight: false }];
    }

    const normalizedQuery = query.toLowerCase();
    const normalizedText = text.toLowerCase();
    const result: Array<{ text: string; highlight: boolean }> = [];

    let lastIndex = 0;
    let index = normalizedText.indexOf(normalizedQuery);

    while (index !== -1) {
      // Add non-matching part before this match
      if (index > lastIndex) {
        result.push({
          text: text.slice(lastIndex, index),
          highlight: false,
        });
      }

      // Add matching part (using original case from text)
      result.push({
        text: text.slice(index, index + query.length),
        highlight: true,
      });

      lastIndex = index + query.length;
      index = normalizedText.indexOf(normalizedQuery, lastIndex);
    }

    // Add remaining non-matching text
    if (lastIndex < text.length) {
      result.push({
        text: text.slice(lastIndex),
        highlight: false,
      });
    }

    return result;
  }, [text, query]);

  return (
    <span className={className}>
      {parts.map((part, i) =>
        part.highlight ? (
          <mark
            key={i}
            className={cn(
              'bg-yellow-200 dark:bg-yellow-500/30',
              'text-gray-900 dark:text-yellow-200',
              'rounded px-0.5',
              highlightClassName
            )}
          >
            {part.text}
          </mark>
        ) : (
          <span key={i}>{part.text}</span>
        )
      )}
    </span>
  );
};

export default SearchHighlight;
