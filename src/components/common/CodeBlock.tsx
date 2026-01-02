/**
 * CodeBlock component with syntax highlighting using prism-react-renderer
 */
import { useState } from 'react';
import { Highlight, themes } from 'prism-react-renderer';
import { Copy, Check } from 'lucide-react';
import { cn } from '@/utils/cn';
import { useAppState } from '@/context/AppContext';

type Language = 'python' | 'javascript' | 'typescript' | 'pseudocode';

interface CodeBlockProps {
  code: string;
  language?: Language;
  showLineNumbers?: boolean;
  className?: string;
}

// Map our language types to Prism language types
const languageMap: Record<Language, string> = {
  python: 'python',
  javascript: 'javascript',
  typescript: 'typescript',
  pseudocode: 'javascript', // Use JS highlighting for pseudocode
};

export const CodeBlock: React.FC<CodeBlockProps> = ({
  code,
  language = 'python',
  showLineNumbers = true,
  className,
}) => {
  const [copied, setCopied] = useState(false);
  const { theme } = useAppState();

  // Determine if dark mode is active
  const isDark =
    theme === 'dark' ||
    (theme === 'system' &&
      typeof window !== 'undefined' &&
      window.matchMedia('(prefers-color-scheme: dark)').matches);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy code:', err);
    }
  };

  const prismLanguage = languageMap[language] || 'javascript';
  const codeTheme = isDark ? themes.vsDark : themes.vsLight;

  return (
    <div className={cn('relative group rounded-lg overflow-hidden', className)}>
      {/* Header with language badge and copy button */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-100 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <span className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase">
          {language}
        </span>
        <button
          onClick={handleCopy}
          className={cn(
            'p-1.5 rounded-md transition-colors duration-200',
            'text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200',
            'hover:bg-gray-200 dark:hover:bg-gray-700',
            'focus:outline-none focus:ring-2 focus:ring-blue-500'
          )}
          aria-label={copied ? 'Copied!' : 'Copy code'}
        >
          {copied ? (
            <Check className="w-4 h-4 text-green-500" />
          ) : (
            <Copy className="w-4 h-4" />
          )}
        </button>
      </div>

      {/* Code content with syntax highlighting */}
      <Highlight theme={codeTheme} code={code.trim()} language={prismLanguage}>
        {({ className: highlightClassName, style, tokens, getLineProps, getTokenProps }) => (
          <pre
            className={cn(
              highlightClassName,
              'overflow-x-auto p-4 text-sm',
              'bg-gray-50 dark:bg-gray-900'
            )}
            style={{ ...style, margin: 0, background: undefined }}
          >
            {tokens.map((line, i) => {
              const lineProps = getLineProps({ line, key: i });
              return (
                <div
                  key={i}
                  {...lineProps}
                  className={cn(lineProps.className, 'table-row')}
                >
                  {showLineNumbers && (
                    <span className="table-cell pr-4 text-right select-none text-gray-400 dark:text-gray-600 text-xs">
                      {i + 1}
                    </span>
                  )}
                  <span className="table-cell">
                    {line.map((token, key) => (
                      <span key={key} {...getTokenProps({ token, key })} />
                    ))}
                  </span>
                </div>
              );
            })}
          </pre>
        )}
      </Highlight>
    </div>
  );
};

export default CodeBlock;
