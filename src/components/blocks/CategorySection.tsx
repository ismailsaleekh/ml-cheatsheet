/**
 * CategorySection component - Section header for each category
 */
import React from 'react';
import {
  BookOpen,
  Database,
  Target,
  TrendingUp,
  Shield,
  BarChart,
  GitBranch,
  Layers,
  Brain,
  Settings,
  type LucideIcon,
} from 'lucide-react';
import type { Category } from '@/types/concept';
import { cn } from '@/utils/cn';
import { getCategoryColor } from '@/utils/constants';

// Icon mapping
const ICON_MAP: Record<string, LucideIcon> = {
  BookOpen,
  Database,
  Target,
  TrendingUp,
  Shield,
  BarChart,
  GitBranch,
  Layers,
  Brain,
  Settings,
};

interface CategorySectionProps {
  category: Category;
  children: React.ReactNode;
}

export const CategorySection: React.FC<CategorySectionProps> = ({
  category,
  children,
}) => {
  const colors = getCategoryColor(category.id);
  const Icon = ICON_MAP[category.icon] || BookOpen;

  return (
    <section className="relative">
      {/* Category Header */}
      <div className="mb-4">
        <div className="flex items-center gap-3">
          <div
            className={cn(
              'p-2 rounded-lg',
              colors.bg,
              'text-white'
            )}
          >
            <Icon className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-gray-900 dark:text-white">
              {category.name}
            </h2>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              {category.description}
            </p>
          </div>
        </div>
      </div>

      {/* Category Content */}
      <div className={cn(
        'relative pl-4',
        'before:absolute before:left-0 before:top-0 before:bottom-0',
        'before:w-1 before:rounded-full',
        colors.bg.replace('bg-', 'before:bg-'),
        'before:opacity-20'
      )}>
        {children}
      </div>
    </section>
  );
};

export default CategorySection;
