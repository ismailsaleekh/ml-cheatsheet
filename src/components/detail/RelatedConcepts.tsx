/**
 * RelatedConcepts - Shows related concepts and prerequisites as clickable chips
 */
import { ArrowRight, ArrowLeft } from 'lucide-react';
import { useConcept } from '@/hooks/useConceptData';
import { cn } from '@/utils/cn';

interface RelatedConceptsProps {
  relatedIds: string[];
  prerequisiteIds: string[];
  onNavigate: (conceptId: string) => void;
}

// Component for a single concept chip
const ConceptChip: React.FC<{
  conceptId: string;
  onClick: () => void;
}> = ({ conceptId, onClick }) => {
  const concept = useConcept(conceptId);

  if (!concept) {
    return null;
  }

  return (
    <button
      onClick={onClick}
      className={cn(
        'inline-flex items-center gap-1 px-3 py-1.5 rounded-full',
        'text-sm font-medium',
        'bg-gray-100 dark:bg-gray-800',
        'text-gray-700 dark:text-gray-300',
        'hover:bg-gray-200 dark:hover:bg-gray-700',
        'transition-colors duration-150',
        'focus:outline-none focus:ring-2 focus:ring-blue-500'
      )}
    >
      {concept.name}
    </button>
  );
};

export const RelatedConcepts: React.FC<RelatedConceptsProps> = ({
  relatedIds,
  prerequisiteIds,
  onNavigate,
}) => {
  const hasRelated = relatedIds.length > 0;
  const hasPrerequisites = prerequisiteIds.length > 0;

  if (!hasRelated && !hasPrerequisites) {
    return null;
  }

  return (
    <div className="space-y-4 pt-4 border-t border-gray-200 dark:border-gray-700">
      {/* Prerequisites */}
      {hasPrerequisites && (
        <div>
          <div className="flex items-center gap-2 mb-2">
            <ArrowLeft className="w-4 h-4 text-gray-400" />
            <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400">
              Prerequisites
            </h4>
          </div>
          <div className="flex flex-wrap gap-2">
            {prerequisiteIds.map((id) => (
              <ConceptChip key={id} conceptId={id} onClick={() => onNavigate(id)} />
            ))}
          </div>
        </div>
      )}

      {/* Related concepts */}
      {hasRelated && (
        <div>
          <div className="flex items-center gap-2 mb-2">
            <ArrowRight className="w-4 h-4 text-gray-400" />
            <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400">
              Related Concepts
            </h4>
          </div>
          <div className="flex flex-wrap gap-2">
            {relatedIds.map((id) => (
              <ConceptChip key={id} conceptId={id} onClick={() => onNavigate(id)} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default RelatedConcepts;
