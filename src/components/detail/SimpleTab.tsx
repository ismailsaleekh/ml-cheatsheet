/**
 * SimpleTab - Plain English explanation content
 */
import type { Concept } from '@/types/concept';
import { cn } from '@/utils/cn';

interface SimpleTabProps {
  concept: Concept;
}

export const SimpleTab: React.FC<SimpleTabProps> = ({ concept }) => {
  // Split explanation into paragraphs
  const paragraphs = concept.simpleExplanation.split('\n').filter((p) => p.trim());

  return (
    <div
      role="tabpanel"
      id="panel-simple"
      aria-labelledby="tab-simple"
    >
      <div className="space-y-4">
        {paragraphs.map((paragraph, index) => (
          <p
            key={index}
            className={cn(
              'text-lg text-gray-700 dark:text-gray-300',
              'leading-relaxed'
            )}
          >
            {paragraph}
          </p>
        ))}
      </div>

      {/* Visual hint that this is the simplified version */}
      <div className="mt-6 p-4 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
        <p className="text-sm text-blue-700 dark:text-blue-300">
          This is a simplified explanation. Switch to the <strong>Technical</strong> tab for more details, or <strong>Example</strong> for code samples.
        </p>
      </div>
    </div>
  );
};

export default SimpleTab;
