/**
 * DetailModal - Main modal component for viewing concept details
 * Includes swipe gestures for mobile navigation
 */
import { useEffect, useRef, useCallback, useState } from 'react';
import { motion, AnimatePresence, type PanInfo } from 'framer-motion';
import { Check, Bookmark } from 'lucide-react';
import type { DetailTab } from '@/types/state';
import { useAppState, useAppDispatch } from '@/context/AppContext';
import { useProgressContext } from '@/context/ProgressContext';
import { useConcept, useConceptCategory } from '@/hooks/useConceptData';
import { useModalNavigation } from '@/hooks/useModalNavigation';
import { useFocusTrap } from '@/hooks/useFocusTrap';
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts';
import { useMobileDetect } from '@/hooks/useMobileDetect';
import { cn } from '@/utils/cn';
import { DetailHeader } from './DetailHeader';
import { DetailTabs } from './DetailTabs';
import { TechnicalTab } from './TechnicalTab';
import { SimpleTab } from './SimpleTab';
import { ExampleTab } from './ExampleTab';
import { RelatedConcepts } from './RelatedConcepts';
import { ModalNavigation } from './ModalNavigation';

// Swipe threshold in pixels
const SWIPE_THRESHOLD = 50;

export const DetailModal: React.FC = () => {
  const { selectedConceptId, modalOpen, activeTab } = useAppState();
  const dispatch = useAppDispatch();
  const modalRef = useRef<HTMLDivElement>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const { isTouchDevice } = useMobileDetect();

  // Track swipe state for visual feedback
  const [swipeOffset, setSwipeOffset] = useState(0);

  const concept = useConcept(selectedConceptId);
  const category = useConceptCategory(selectedConceptId);
  const { previousConcept, nextConcept, goToPrevious, goToNext } = useModalNavigation(selectedConceptId);
  const { isLearned, isBookmarked, toggleLearned, toggleBookmark, recordVisit } = useProgressContext();

  // Track if current concept is learned/bookmarked
  const learned = selectedConceptId ? isLearned(selectedConceptId) : false;
  const bookmarked = selectedConceptId ? isBookmarked(selectedConceptId) : false;

  // Focus trap
  useFocusTrap(modalRef, modalOpen);

  // Close modal handler
  const handleClose = useCallback(() => {
    dispatch({ type: 'SET_MODAL_OPEN', payload: false });
  }, [dispatch]);

  // Tab change handler
  const handleTabChange = useCallback((tab: DetailTab) => {
    dispatch({ type: 'SET_TAB', payload: tab });
    if (contentRef.current) {
      contentRef.current.scrollTop = 0;
    }
  }, [dispatch]);

  // Navigate to related concept
  const handleNavigateToConcept = useCallback((conceptId: string) => {
    dispatch({ type: 'SELECT_CONCEPT', payload: conceptId });
    if (contentRef.current) {
      contentRef.current.scrollTop = 0;
    }
  }, [dispatch]);

  // Handle swipe gesture
  const handleDragEnd = useCallback(
    (_event: MouseEvent | TouchEvent | PointerEvent, info: PanInfo) => {
      setSwipeOffset(0);

      // Horizontal swipe for navigation
      if (Math.abs(info.offset.x) > SWIPE_THRESHOLD) {
        if (info.offset.x > 0 && previousConcept) {
          goToPrevious();
        } else if (info.offset.x < 0 && nextConcept) {
          goToNext();
        }
      }
    },
    [previousConcept, nextConcept, goToPrevious, goToNext]
  );

  // Update swipe offset for visual feedback
  const handleDrag = useCallback(
    (_event: MouseEvent | TouchEvent | PointerEvent, info: PanInfo) => {
      // Only update if there's a valid target
      const canGoLeft = previousConcept && info.offset.x > 0;
      const canGoRight = nextConcept && info.offset.x < 0;

      if (canGoLeft || canGoRight) {
        setSwipeOffset(info.offset.x * 0.3); // Damped feedback
      }
    },
    [previousConcept, nextConcept]
  );

  // Keyboard shortcuts
  useKeyboardShortcuts([
    { key: 'Escape', action: handleClose, enabled: modalOpen },
    { key: 'ArrowLeft', action: goToPrevious, enabled: modalOpen && !!previousConcept },
    { key: 'ArrowRight', action: goToNext, enabled: modalOpen && !!nextConcept },
    { key: '1', action: () => handleTabChange('technical'), enabled: modalOpen },
    { key: '2', action: () => handleTabChange('simple'), enabled: modalOpen },
    { key: '3', action: () => handleTabChange('example'), enabled: modalOpen },
  ]);

  // Lock body scroll when modal is open
  useEffect(() => {
    if (modalOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [modalOpen]);

  // Reset scroll when concept changes
  useEffect(() => {
    if (contentRef.current) {
      contentRef.current.scrollTop = 0;
    }
    setSwipeOffset(0);
  }, [selectedConceptId]);

  // Record visit when concept is viewed
  useEffect(() => {
    if (selectedConceptId && modalOpen) {
      recordVisit(selectedConceptId);
    }
  }, [selectedConceptId, modalOpen, recordVisit]);

  // Handle learn/bookmark toggles
  const handleLearnToggle = useCallback(() => {
    if (selectedConceptId) {
      toggleLearned(selectedConceptId);
    }
  }, [selectedConceptId, toggleLearned]);

  const handleBookmarkToggle = useCallback(() => {
    if (selectedConceptId) {
      toggleBookmark(selectedConceptId);
    }
  }, [selectedConceptId, toggleBookmark]);

  if (!concept) {
    return null;
  }

  const renderTabContent = () => {
    switch (activeTab) {
      case 'technical':
        return <TechnicalTab concept={concept} />;
      case 'simple':
        return <SimpleTab concept={concept} />;
      case 'example':
        return <ExampleTab concept={concept} />;
      default:
        return <TechnicalTab concept={concept} />;
    }
  };

  return (
    <AnimatePresence>
      {modalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          {/* Backdrop overlay */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="absolute inset-0 bg-black/50 backdrop-blur-sm"
            onClick={handleClose}
            aria-hidden="true"
          />

          {/* Modal content with swipe gesture */}
          <motion.div
            ref={modalRef}
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{
              opacity: 1,
              scale: 1,
              y: 0,
              x: swipeOffset,
            }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ duration: 0.2, ease: 'easeOut' }}
            drag={isTouchDevice ? 'x' : false}
            dragConstraints={{ left: 0, right: 0 }}
            dragElastic={0.1}
            onDrag={handleDrag}
            onDragEnd={handleDragEnd}
            className={cn(
              'relative w-full max-w-4xl max-h-[90vh]',
              'bg-white dark:bg-gray-900',
              'rounded-2xl shadow-2xl',
              'flex flex-col overflow-hidden',
              // Mobile: Full screen
              'sm:max-h-[90vh]',
              // Touch device visual hints
              isTouchDevice && 'cursor-grab active:cursor-grabbing'
            )}
            role="dialog"
            aria-modal="true"
            aria-labelledby="modal-title"
          >
            {/* Swipe indicators */}
            {isTouchDevice && (
              <>
                {previousConcept && swipeOffset > 20 && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: Math.min(swipeOffset / 100, 0.5) }}
                    className="absolute left-2 top-1/2 -translate-y-1/2 z-10 text-gray-400"
                  >
                    <span className="text-2xl">←</span>
                  </motion.div>
                )}
                {nextConcept && swipeOffset < -20 && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: Math.min(Math.abs(swipeOffset) / 100, 0.5) }}
                    className="absolute right-2 top-1/2 -translate-y-1/2 z-10 text-gray-400"
                  >
                    <span className="text-2xl">→</span>
                  </motion.div>
                )}
              </>
            )}

            {/* Header */}
            <DetailHeader
              concept={concept}
              category={category}
              onClose={handleClose}
            />

            {/* Tabs */}
            <DetailTabs activeTab={activeTab} onTabChange={handleTabChange} />

            {/* Scrollable content */}
            <div
              ref={contentRef}
              className="flex-1 overflow-y-auto px-6 py-4"
            >
              {/* Tab content */}
              <motion.div
                key={activeTab}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.2 }}
              >
                {renderTabContent()}
              </motion.div>

              {/* Related concepts */}
              <div className="mt-6">
                <RelatedConcepts
                  relatedIds={concept.relatedConcepts}
                  prerequisiteIds={concept.prerequisites}
                  onNavigate={handleNavigateToConcept}
                />
              </div>
            </div>

            {/* Navigation footer */}
            <div className="px-6 py-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
              {/* Action buttons */}
              <div className="flex items-center justify-center gap-3 mb-4">
                <button
                  onClick={handleLearnToggle}
                  className={cn(
                    'flex items-center gap-2 px-4 py-2 rounded-lg transition-all',
                    learned
                      ? 'bg-green-500 text-white hover:bg-green-600'
                      : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
                  )}
                >
                  <Check className="w-4 h-4" />
                  <span className="text-sm font-medium">
                    {learned ? 'Learned' : 'Mark as Learned'}
                  </span>
                </button>
                <button
                  onClick={handleBookmarkToggle}
                  className={cn(
                    'flex items-center gap-2 px-4 py-2 rounded-lg transition-all',
                    bookmarked
                      ? 'bg-yellow-500 text-white hover:bg-yellow-600'
                      : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
                  )}
                >
                  <Bookmark className={cn('w-4 h-4', bookmarked && 'fill-current')} />
                  <span className="text-sm font-medium">
                    {bookmarked ? 'Bookmarked' : 'Bookmark'}
                  </span>
                </button>
              </div>

              <ModalNavigation
                previousConcept={previousConcept}
                nextConcept={nextConcept}
                onPrevious={goToPrevious}
                onNext={goToNext}
              />
              {/* Mobile swipe hint */}
              {isTouchDevice && (previousConcept || nextConcept) && (
                <p className="text-center text-xs text-gray-400 mt-2">
                  Swipe left or right to navigate
                </p>
              )}
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};

export default DetailModal;
