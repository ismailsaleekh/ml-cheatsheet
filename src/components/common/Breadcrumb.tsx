/**
 * Breadcrumb navigation component
 */
import { Home, ChevronRight } from 'lucide-react';
import { cn } from '@/utils/cn';
import { useConceptPath, useConceptCategory } from '@/hooks/useConceptData';
import type { Concept } from '@/types/concept';

interface BreadcrumbProps {
  conceptId: string | null;
  onNavigate?: (conceptId: string | null) => void;
  className?: string;
}

export const Breadcrumb: React.FC<BreadcrumbProps> = ({
  conceptId,
  onNavigate,
  className,
}) => {
  const path = useConceptPath(conceptId);
  const category = useConceptCategory(conceptId);

  if (!conceptId || path.length === 0) {
    return null;
  }

  const handleHomeClick = () => {
    onNavigate?.(null);
  };

  const handleItemClick = (concept: Concept) => {
    onNavigate?.(concept.id);
  };

  // Build breadcrumb items: Home > Category > ...path
  const items: Array<{ label: string; id: string | null; isCategory?: boolean }> = [
    { label: 'Home', id: null },
  ];

  if (category) {
    items.push({ label: category.name, id: category.id, isCategory: true });
  }

  path.forEach((concept) => {
    items.push({ label: concept.name, id: concept.id });
  });

  // For mobile, show collapsed version if too many items
  const maxVisibleItems = 4;
  const shouldCollapse = items.length > maxVisibleItems;
  const visibleItems = shouldCollapse
    ? [
        items[0], // Home
        { label: '...', id: null }, // Ellipsis
        ...items.slice(-2), // Last 2 items
      ]
    : items;

  return (
    <nav className={cn('flex items-center text-sm', className)} aria-label="Breadcrumb">
      <ol className="flex items-center space-x-1 overflow-x-auto">
        {visibleItems.map((item, index) => {
          const isLast = index === visibleItems.length - 1;
          const isEllipsis = item.label === '...';

          return (
            <li key={index} className="flex items-center">
              {index > 0 && (
                <ChevronRight className="w-4 h-4 text-gray-400 dark:text-gray-500 flex-shrink-0 mx-1" />
              )}
              {isEllipsis ? (
                <span className="text-gray-400 dark:text-gray-500 px-1">...</span>
              ) : isLast ? (
                <span
                  className={cn(
                    'px-2 py-1 rounded-md',
                    'text-gray-900 dark:text-white font-medium',
                    'bg-gray-100 dark:bg-gray-800'
                  )}
                  aria-current="page"
                >
                  {item.label}
                </span>
              ) : (
                <button
                  onClick={() => {
                    if (item.id === null) {
                      handleHomeClick();
                    } else if (!item.isCategory) {
                      const concept = path.find((c) => c.id === item.id);
                      if (concept) {
                        handleItemClick(concept);
                      }
                    }
                  }}
                  className={cn(
                    'flex items-center gap-1 px-2 py-1 rounded-md',
                    'text-gray-600 dark:text-gray-400',
                    'hover:text-gray-900 dark:hover:text-white',
                    'hover:bg-gray-100 dark:hover:bg-gray-800',
                    'transition-colors duration-150',
                    'focus:outline-none focus:ring-2 focus:ring-blue-500'
                  )}
                >
                  {item.id === null && <Home className="w-4 h-4" />}
                  <span className="whitespace-nowrap">{item.id === null ? '' : item.label}</span>
                </button>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
};

export default Breadcrumb;
