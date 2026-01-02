/**
 * BlockContainer component - Main container for displaying all blocks
 */
import { AnimatePresence } from 'framer-motion';
import { useAppState, useAppDispatch } from '@/context/AppContext';
import {
  useCategories,
  useConceptsByCategory,
  useHasChildren,
} from '@/hooks/useConceptData';
import { Block } from './Block';
import { BlockChildren } from './BlockChildren';
import { CategorySection } from './CategorySection';

export const BlockContainer: React.FC = () => {
  const categories = useCategories();
  const conceptsByCategory = useConceptsByCategory();

  return (
    <div className="space-y-8">
      {categories.map((category) => {
        const concepts = conceptsByCategory.get(category.id) || [];

        return (
          <CategorySection key={category.id} category={category}>
            <div className="space-y-4">
              {concepts.map((concept) => (
                <ConceptBlockWithChildren
                  key={concept.id}
                  conceptId={concept.id}
                  categoryId={category.id}
                />
              ))}
            </div>
          </CategorySection>
        );
      })}
    </div>
  );
};

/**
 * Individual concept block with recursive children
 */
interface ConceptBlockWithChildrenProps {
  conceptId: string;
  categoryId: string;
}

const ConceptBlockWithChildren: React.FC<ConceptBlockWithChildrenProps> = ({
  conceptId,
  categoryId,
}) => {
  const { concepts, expandedIds } = useAppState();
  const dispatch = useAppDispatch();
  const hasChildConcepts = useHasChildren(conceptId);

  const concept = concepts.find((c) => c.id === conceptId);
  if (!concept) return null;

  const isExpanded = expandedIds.has(conceptId);

  const handleToggle = () => {
    dispatch({ type: 'TOGGLE_EXPAND', payload: conceptId });
  };

  const handleViewDetails = () => {
    dispatch({ type: 'SELECT_CONCEPT', payload: conceptId });
  };

  return (
    <div>
      <Block
        concept={concept}
        categoryId={categoryId}
        isExpanded={isExpanded}
        hasChildren={hasChildConcepts}
        onToggle={handleToggle}
        onViewDetails={handleViewDetails}
        depth={0}
      />
      <AnimatePresence>
        {isExpanded && hasChildConcepts && (
          <BlockChildren
            parentId={conceptId}
            categoryId={categoryId}
            depth={1}
          />
        )}
      </AnimatePresence>
    </div>
  );
};

export default BlockContainer;
