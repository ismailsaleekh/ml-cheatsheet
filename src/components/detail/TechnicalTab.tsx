/**
 * TechnicalTab - Full technical explanation content
 */
import type { Concept } from '@/types/concept';
import { cn } from '@/utils/cn';

interface TechnicalTabProps {
  concept: Concept;
}

export const TechnicalTab: React.FC<TechnicalTabProps> = ({ concept }) => {
  // Split explanation into paragraphs for better readability
  const paragraphs = concept.fullExplanation.split('\n').filter((p) => p.trim());

  return (
    <div
      role="tabpanel"
      id="panel-technical"
      aria-labelledby="tab-technical"
      className="prose prose-gray dark:prose-invert max-w-none"
    >
      <div className="space-y-4">
        {paragraphs.map((paragraph, index) => (
          <p
            key={index}
            className={cn(
              'text-gray-700 dark:text-gray-300',
              'leading-relaxed'
            )}
          >
            {paragraph}
          </p>
        ))}
      </div>
    </div>
  );
};

export default TechnicalTab;
