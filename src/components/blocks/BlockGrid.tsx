/**
 * BlockGrid component - Responsive grid layout for blocks
 */
import React from 'react';
import { motion } from 'framer-motion';
import { cn } from '@/utils/cn';
import { STAGGER_CONTAINER, STAGGER_ITEM } from '@/utils/constants';

interface BlockGridProps {
  children: React.ReactNode;
  className?: string;
}

export const BlockGrid: React.FC<BlockGridProps> = ({ children, className }) => {
  return (
    <motion.div
      variants={STAGGER_CONTAINER}
      initial="hidden"
      animate="show"
      className={cn(
        'grid gap-4',
        'grid-cols-1',
        'sm:grid-cols-2',
        'lg:grid-cols-3',
        'xl:grid-cols-4',
        className
      )}
    >
      {children}
    </motion.div>
  );
};

/**
 * BlockGridItem wrapper for staggered animation
 */
interface BlockGridItemProps {
  children: React.ReactNode;
  className?: string;
}

export const BlockGridItem: React.FC<BlockGridItemProps> = ({ children, className }) => {
  return (
    <motion.div variants={STAGGER_ITEM} className={className}>
      {children}
    </motion.div>
  );
};

export default BlockGrid;
