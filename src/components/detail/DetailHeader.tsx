/**
 * DetailHeader - Modal header with concept info and close button
 */
import { X } from 'lucide-react';
import type { Concept, Category } from '@/types/concept';
import { Badge } from '@/components/common/Badge';
import { cn } from '@/utils/cn';
import { getCategoryColor } from '@/utils/constants';

interface DetailHeaderProps {
  concept: Concept;
  category?: Category;
  onClose: () => void;
}

export const DetailHeader: React.FC<DetailHeaderProps> = ({
  concept,
  category,
  onClose,
}) => {
  const categoryColors = category ? getCategoryColor(category.id) : null;

  return (
    <div className="relative px-6 pt-6 pb-4 border-b border-gray-200 dark:border-gray-700">
      {/* Close button */}
      <button
        onClick={onClose}
        className={cn(
          'absolute top-4 right-4 p-2 rounded-full',
          'text-gray-400 hover:text-gray-600 dark:hover:text-gray-300',
          'hover:bg-gray-100 dark:hover:bg-gray-700',
          'transition-colors duration-150',
          'focus:outline-none focus:ring-2 focus:ring-blue-500'
        )}
        aria-label="Close modal"
      >
        <X className="w-5 h-5" />
      </button>

      {/* Category indicator */}
      {category && categoryColors && (
        <div
          className={cn(
            'inline-flex items-center gap-2 px-3 py-1 rounded-full mb-3',
            categoryColors.bgLight
          )}
        >
          <div className={cn('w-2 h-2 rounded-full', categoryColors.bg)} />
          <span className={cn('text-xs font-medium', categoryColors.text)}>
            {category.name}
          </span>
        </div>
      )}

      {/* Concept name */}
      <h2
        id="modal-title"
        className="text-2xl font-bold text-gray-900 dark:text-white pr-10"
      >
        {concept.name}
      </h2>

      {/* Badges */}
      <div className="flex flex-wrap gap-2 mt-3">
        <Badge variant="section" categoryId={category?.id}>
          {concept.sectionId}
        </Badge>
        <Badge variant="difficulty" difficulty={concept.difficulty}>
          {concept.difficulty}
        </Badge>
        {concept.tags.slice(0, 3).map((tag) => (
          <Badge key={tag} variant="tag">
            {tag}
          </Badge>
        ))}
      </div>
    </div>
  );
};

export default DetailHeader;
