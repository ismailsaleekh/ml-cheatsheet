/**
 * EmptyState - Reusable empty state component
 */
import { cn } from '@/utils/cn';
import type { LucideIcon } from 'lucide-react';

interface EmptyStateProps {
  icon?: LucideIcon;
  title: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
  className?: string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  icon: Icon,
  title,
  description,
  action,
  className,
}) => {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center py-12 px-4',
        'text-center',
        className
      )}
    >
      {Icon && (
        <Icon
          className={cn(
            'w-12 h-12 mb-4',
            'text-gray-300 dark:text-gray-600'
          )}
        />
      )}
      <h3 className="text-lg font-medium text-gray-900 dark:text-white">
        {title}
      </h3>
      {description && (
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400 max-w-sm">
          {description}
        </p>
      )}
      {action && (
        <button
          onClick={action.onClick}
          className={cn(
            'mt-4 px-4 py-2 rounded-lg',
            'bg-blue-500 hover:bg-blue-600',
            'text-white text-sm font-medium',
            'transition-colors'
          )}
        >
          {action.label}
        </button>
      )}
    </div>
  );
};

export default EmptyState;
