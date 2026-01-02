/**
 * BlockChildren component - Renders children of an expanded block
 */
import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { Concept } from '@/types/concept';
import { Block } from './Block';
import { useAppState, useAppDispatch } from '@/context/AppContext';
import { useConceptChildren, useHasChildren } from '@/hooks/useConceptData';
import { cn } from '@/utils/cn';

interface BlockChildrenProps {
  parentId: string;
  categoryId: string;
  depth: number;
}

export const BlockChildren: React.FC<BlockChildrenProps> = ({
  parentId,
  categoryId,
  depth,
}) => {
  const children = useConceptChildren(parentId);
  const { expandedIds } = useAppState();
  const dispatch = useAppDispatch();

  if (children.length === 0) {
    return null;
  }

  return (
    <motion.div
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: 'auto' }}
      exit={{ opacity: 0, height: 0 }}
      transition={{ duration: 0.3, ease: 'easeOut' }}
      className="overflow-hidden"
    >
      <div className={cn('mt-3 space-y-3', depth > 0 && 'pl-4')}>
        {children.map((child) => (
          <BlockWithChildren
            key={child.id}
            concept={child}
            categoryId={categoryId}
            depth={depth}
            isExpanded={expandedIds.has(child.id)}
            onToggle={() => dispatch({ type: 'TOGGLE_EXPAND', payload: child.id })}
            onViewDetails={() => dispatch({ type: 'SELECT_CONCEPT', payload: child.id })}
          />
        ))}
      </div>
    </motion.div>
  );
};

/**
 * Block with recursive children rendering
 */
interface BlockWithChildrenProps {
  concept: Concept;
  categoryId: string;
  depth: number;
  isExpanded: boolean;
  onToggle: () => void;
  onViewDetails: () => void;
}

const BlockWithChildren: React.FC<BlockWithChildrenProps> = ({
  concept,
  categoryId,
  depth,
  isExpanded,
  onToggle,
  onViewDetails,
}) => {
  const hasChildConcepts = useHasChildren(concept.id);

  return (
    <div>
      <Block
        concept={concept}
        categoryId={categoryId}
        isExpanded={isExpanded}
        hasChildren={hasChildConcepts}
        onToggle={onToggle}
        onViewDetails={onViewDetails}
        depth={depth}
      />
      <AnimatePresence>
        {isExpanded && hasChildConcepts && (
          <BlockChildren
            parentId={concept.id}
            categoryId={categoryId}
            depth={depth + 1}
          />
        )}
      </AnimatePresence>
    </div>
  );
};

export default BlockChildren;
