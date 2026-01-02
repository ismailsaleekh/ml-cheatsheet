/**
 * Reusable Badge component for displaying labels, tags, and status indicators
 */
import type { Difficulty } from '@/types/concept';
import { cn } from '@/utils/cn';
import { getCategoryColor, getDifficultyColor } from '@/utils/constants';

type BadgeVariant = 'section' | 'difficulty' | 'category' | 'tag' | 'default';

interface BadgeProps {
  variant?: BadgeVariant;
  children: React.ReactNode;
  categoryId?: string;
  difficulty?: Difficulty;
  className?: string;
}

export const Badge: React.FC<BadgeProps> = ({
  variant = 'default',
  children,
  categoryId,
  difficulty,
  className,
}) => {
  const getVariantClasses = () => {
    switch (variant) {
      case 'section':
        if (categoryId) {
          const colors = getCategoryColor(categoryId);
          return cn(colors.bgLight, colors.text);
        }
        return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400';

      case 'difficulty':
        if (difficulty) {
          return getDifficultyColor(difficulty);
        }
        return 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300';

      case 'category':
        if (categoryId) {
          const colors = getCategoryColor(categoryId);
          return cn(colors.bgLight, colors.text);
        }
        return 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300';

      case 'tag':
        return 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300';

      default:
        return 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300';
    }
  };

  return (
    <span
      className={cn(
        'inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium',
        getVariantClasses(),
        className
      )}
    >
      {children}
    </span>
  );
};

export default Badge;
