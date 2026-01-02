/**
 * ExampleTab - Code examples, tables, and visuals
 */
import type { Concept } from '@/types/concept';
import { CodeBlock } from '@/components/common/CodeBlock';
import { cn } from '@/utils/cn';

interface ExampleTabProps {
  concept: Concept;
}

export const ExampleTab: React.FC<ExampleTabProps> = ({ concept }) => {
  const { example } = concept;

  return (
    <div
      role="tabpanel"
      id="panel-example"
      aria-labelledby="tab-example"
      className="space-y-6"
    >
      {/* Description */}
      {example.description && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
            Description
          </h4>
          <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
            {example.description}
          </p>
        </div>
      )}

      {/* Code block */}
      {example.code && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
            Code Example
          </h4>
          <CodeBlock
            code={example.code}
            language={example.codeLanguage || 'python'}
            showLineNumbers={true}
          />
        </div>
      )}

      {/* Table */}
      {example.table && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
            Comparison
          </h4>
          <div className="overflow-x-auto rounded-lg border border-gray-200 dark:border-gray-700">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-800">
                <tr>
                  {example.table.headers.map((header, index) => (
                    <th
                      key={index}
                      className={cn(
                        'px-4 py-3 text-left text-xs font-medium uppercase tracking-wider',
                        'text-gray-500 dark:text-gray-400'
                      )}
                    >
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
                {example.table.rows.map((row, rowIndex) => (
                  <tr key={rowIndex}>
                    {row.map((cell, cellIndex) => (
                      <td
                        key={cellIndex}
                        className={cn(
                          'px-4 py-3 text-sm',
                          'text-gray-700 dark:text-gray-300',
                          cellIndex === 0 && 'font-medium'
                        )}
                      >
                        {cell}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Visual/Image placeholder */}
      {example.visual && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
            Visual
          </h4>
          <div className="p-4 rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
            <p className="text-sm text-gray-500 dark:text-gray-400 italic">
              Visual: {example.visual}
            </p>
          </div>
        </div>
      )}

      {/* Empty state */}
      {!example.code && !example.table && !example.visual && (
        <div className="p-6 text-center rounded-lg border border-dashed border-gray-300 dark:border-gray-700">
          <p className="text-gray-500 dark:text-gray-400">
            No code examples available for this concept yet.
          </p>
        </div>
      )}
    </div>
  );
};

export default ExampleTab;
