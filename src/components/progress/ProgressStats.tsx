/**
 * ProgressStats - Display learning statistics
 */
import React from 'react';
import { Flame, Trophy, BookCheck, Target } from 'lucide-react';
import { cn } from '@/utils/cn';
import type { StudyStats } from '@/types/progress';

interface ProgressStatsProps {
  stats: StudyStats;
  totalConcepts: number;
  compact?: boolean;
  className?: string;
}

interface StatItemProps {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  color: string;
  compact?: boolean;
}

const StatItem: React.FC<StatItemProps> = ({ icon, label, value, color, compact }) => (
  <div
    className={cn(
      'flex items-center gap-2',
      compact ? 'flex-row' : 'flex-col',
      compact && 'justify-between'
    )}
  >
    <div className="flex items-center gap-2">
      <div
        className={cn(
          'p-1.5 rounded-lg',
          color
        )}
      >
        {icon}
      </div>
      {compact && <span className="text-sm text-gray-600 dark:text-gray-400">{label}</span>}
    </div>
    <div className={cn('text-center', compact && 'text-right')}>
      <div className="text-lg font-bold text-gray-900 dark:text-white">{value}</div>
      {!compact && (
        <div className="text-xs text-gray-500 dark:text-gray-400">{label}</div>
      )}
    </div>
  </div>
);

export const ProgressStats: React.FC<ProgressStatsProps> = ({
  stats,
  totalConcepts,
  compact = false,
  className,
}) => {
  const percentage = totalConcepts > 0
    ? Math.round((stats.totalLearned / totalConcepts) * 100)
    : 0;

  return (
    <div
      className={cn(
        compact
          ? 'space-y-3'
          : 'grid grid-cols-2 md:grid-cols-4 gap-4',
        className
      )}
    >
      <StatItem
        icon={<BookCheck className="w-4 h-4 text-green-600" />}
        label="Learned"
        value={`${stats.totalLearned}/${totalConcepts}`}
        color="bg-green-100 dark:bg-green-900/30"
        compact={compact}
      />
      <StatItem
        icon={<Target className="w-4 h-4 text-blue-600" />}
        label="Progress"
        value={`${percentage}%`}
        color="bg-blue-100 dark:bg-blue-900/30"
        compact={compact}
      />
      <StatItem
        icon={<Flame className="w-4 h-4 text-orange-600" />}
        label="Current Streak"
        value={`${stats.currentStreak} days`}
        color="bg-orange-100 dark:bg-orange-900/30"
        compact={compact}
      />
      <StatItem
        icon={<Trophy className="w-4 h-4 text-yellow-600" />}
        label="Best Streak"
        value={`${stats.longestStreak} days`}
        color="bg-yellow-100 dark:bg-yellow-900/30"
        compact={compact}
      />
    </div>
  );
};

export default ProgressStats;
