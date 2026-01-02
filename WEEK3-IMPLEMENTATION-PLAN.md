# Week 3: Search & Polish - Implementation Plan

## Overview

**Goal**: Implement real-time search functionality with fuzzy matching, search results dropdown, mobile optimizations, and polish animations/interactions.

**Deliverable**: Complete, polished application with functional search across all concepts.

---

## Week 3 Tasks from Project Plan

| # | Task | Description | Status |
|---|------|-------------|--------|
| 1 | Implement search functionality | Real-time filtering across concepts | To Do |
| 2 | Add search results dropdown | Categorized results with highlighting | To Do |
| 3 | Implement dark/light theme toggle | User theme preference | ✅ Done (Week 1) |
| 4 | Add responsive design adjustments | Mobile/tablet layouts | To Do |
| 5 | Mobile touch interactions | Swipe gestures, touch-friendly UI | To Do |
| 6 | Loading and error states | Skeleton loaders, error handling | ✅ Done (Week 1) |

---

## Detailed Requirements Analysis

### 3.1 Search Functionality

**From Project Plan (Section 3.4):**

```
┌─────────────────────────────────────┐
│  🔍  "gradient des"                 │
├─────────────────────────────────────┤
│  OPTIMIZATION                       │
│  ├─ Gradient Descent               │
│  ├─ Stochastic Gradient Descent    │
│  └─ Projected Gradient Descent     │
│  REGULARIZATION                     │
│  └─ Gradient Clipping              │
│                                     │
│  Press Enter to view first result   │
└─────────────────────────────────────┘
```

**Search Behavior:**
1. Real-time filtering: Results update as user types (debounced 200ms)
2. Search scope: Name, tags, explanations
3. Fuzzy matching: Tolerate typos (using Fuse.js)
4. Results display: Dropdown with categorized results
5. Keyboard navigation: Arrow keys to navigate, Enter to select

**Search Algorithm:**
- Primary: Exact substring match in name (highest priority)
- Secondary: Tag matching
- Tertiary: Content matching (fullExplanation, simpleExplanation)
- Fuzzy: Handle typos with configurable threshold

### 3.2 Keyboard Shortcuts

**From Project Plan (Section 5.2) - Search Related:**

| Shortcut | Action |
|----------|--------|
| `/` or `Ctrl+K` | Focus search input |
| `Escape` | Clear search / Close dropdown |
| `↑` `↓` | Navigate search results |
| `Enter` | Select highlighted result |

### 3.3 Mobile Requirements

**Touch Gestures (Section 5.3):**

| Gesture | Action |
|---------|--------|
| Tap block | Expand/collapse |
| Tap info icon | Open modal |
| Swipe left/right (modal) | Previous/Next concept |
| Pull down (modal) | Close modal |

**Mobile Layout:**
- Full-screen search overlay on mobile
- Touch-friendly tap targets (min 44px)
- Bottom sheet for search on mobile
- Collapsible header on scroll

---

## Current State Analysis

### Already Implemented

**State & Actions (AppContext.tsx):**
```typescript
// State
searchQuery: string;
searchResults: Concept[];
isSearching: boolean;

// Actions
| { type: 'SET_SEARCH'; payload: string }
| { type: 'SET_SEARCH_RESULTS'; payload: Concept[] }
| { type: 'CLEAR_SEARCH' }
```

**Constants (constants.ts):**
```typescript
SEARCH_DEBOUNCE_MS = 200;
MAX_SEARCH_RESULTS = 20;
```

**Header.tsx:**
- Placeholder search input (currently disabled)
- Already has Search icon from lucide-react

### Needs Implementation

1. **SearchInput component** - Interactive search with focus states
2. **SearchResults component** - Dropdown with grouped results
3. **useSearch hook** - Debounced search logic
4. **searchHelpers.ts** - Search algorithm implementation
5. **Mobile search overlay** - Full-screen search on mobile
6. **Result highlighting** - Highlight matching text
7. **Keyboard navigation** - Arrow keys, Enter, Escape
8. **Mobile touch gestures** - Swipe for modal navigation

---

## New Files to Create

```
src/
├── components/
│   ├── search/
│   │   ├── SearchInput.tsx          # Main search input component
│   │   ├── SearchResults.tsx        # Dropdown results list
│   │   ├── SearchResultItem.tsx     # Individual result item
│   │   ├── SearchHighlight.tsx      # Text with highlighted matches
│   │   ├── MobileSearchOverlay.tsx  # Full-screen mobile search
│   │   └── index.ts
│   └── common/
│       └── EmptyState.tsx           # Reusable empty/no results state
├── hooks/
│   ├── useSearch.ts                 # Search with debouncing
│   ├── useDebounce.ts               # Debounce utility hook
│   └── useMobileDetect.ts           # Mobile device detection
└── utils/
    └── searchHelpers.ts             # Search algorithm functions
```

---

## Implementation Tasks Breakdown

### Task 1: Search Utilities & Hooks

#### 1.1 useDebounce Hook
```typescript
// src/hooks/useDebounce.ts
function useDebounce<T>(value: T, delay: number): T
```
- Generic debounce hook for any value
- Used to debounce search query

#### 1.2 Search Helper Functions
```typescript
// src/utils/searchHelpers.ts

interface SearchResult {
  concept: Concept;
  score: number;
  matchedField: 'name' | 'tags' | 'content';
  matchRanges: Array<{ start: number; end: number }>;
}

function searchConcepts(
  concepts: Concept[],
  query: string,
  options?: SearchOptions
): SearchResult[]

function highlightMatches(
  text: string,
  query: string
): React.ReactNode

function groupResultsByCategory(
  results: SearchResult[],
  categories: Category[]
): Map<Category, SearchResult[]>
```

**Search Algorithm:**
1. Normalize query (lowercase, trim)
2. If query < 2 chars, return empty
3. Score each concept:
   - Name contains query: score = 100
   - Name starts with query word: score = 80
   - Tag exact match: score = 60
   - Tag contains: score = 40
   - Content contains: score = 20
4. Sort by score descending
5. Limit to MAX_SEARCH_RESULTS

#### 1.3 useSearch Hook
```typescript
// src/hooks/useSearch.ts
interface UseSearchResult {
  query: string;
  setQuery: (query: string) => void;
  results: SearchResult[];
  isSearching: boolean;
  selectedIndex: number;
  setSelectedIndex: (index: number) => void;
  clearSearch: () => void;
  selectResult: (index?: number) => void;
}

function useSearch(): UseSearchResult
```

**Features:**
- Debounced search (200ms)
- Keyboard navigation state
- Integration with AppContext
- Auto-expand ancestors when selecting result

---

### Task 2: Search Components

#### 2.1 SearchInput
```typescript
// src/components/search/SearchInput.tsx
interface SearchInputProps {
  placeholder?: string;
  autoFocus?: boolean;
  onFocus?: () => void;
  onBlur?: () => void;
}
```

**Features:**
- Controlled input with state from useSearch
- Clear button (X) when has value
- Loading spinner while searching
- Focus styling
- Keyboard shortcut hint (⌘K)

**Keyboard Handling:**
- Escape: Clear search or blur
- ArrowDown: Focus first result
- Enter: Select first result (if no results selected)

#### 2.2 SearchResults
```typescript
// src/components/search/SearchResults.tsx
interface SearchResultsProps {
  results: SearchResult[];
  selectedIndex: number;
  onSelect: (conceptId: string) => void;
  onHover: (index: number) => void;
  groupByCategory?: boolean;
}
```

**Layout:**
```
┌─────────────────────────────────────┐
│  6 results for "gradient"           │
├─────────────────────────────────────┤
│  OPTIMIZATION                       │
│  ├─ [★] Gradient Descent           │
│  │      Core optimization method    │
│  ├─ Stochastic Gradient Descent    │
│  │      Mini-batch training         │
│  └─ Learning Rate                   │
│      Controls step size             │
│─────────────────────────────────────│
│  FOUNDATIONS                        │
│  └─ Gradient                        │
│      Vector of partial derivatives  │
├─────────────────────────────────────┤
│  ↑↓ Navigate  ⏎ Select  ⎋ Close    │
└─────────────────────────────────────┘
```

**Features:**
- Grouped by category
- Highlighted matching text
- Selected item styling
- Category headers
- Result count
- Keyboard hints footer
- Empty state message
- Scroll into view for selected

#### 2.3 SearchResultItem
```typescript
// src/components/search/SearchResultItem.tsx
interface SearchResultItemProps {
  result: SearchResult;
  isSelected: boolean;
  onClick: () => void;
  onMouseEnter: () => void;
}
```

**Displays:**
- Concept name with highlighted matches
- Brief description (truncated)
- Section ID badge
- Difficulty badge
- Category color indicator

#### 2.4 SearchHighlight
```typescript
// src/components/search/SearchHighlight.tsx
interface SearchHighlightProps {
  text: string;
  query: string;
  className?: string;
}
```
- Wraps matching text in `<mark>` tags
- Case-insensitive matching
- Handles multiple matches

---

### Task 3: Mobile Search Experience

#### 3.1 MobileSearchOverlay
```typescript
// src/components/search/MobileSearchOverlay.tsx
interface MobileSearchOverlayProps {
  isOpen: boolean;
  onClose: () => void;
}
```

**Mobile Layout:**
```
┌─────────────────────────────────────┐
│  ←  Search Concepts            [X]  │
├─────────────────────────────────────┤
│  ┌─────────────────────────────┐   │
│  │ 🔍  Type to search...       │   │
│  └─────────────────────────────┘   │
│                                     │
│  RECENT SEARCHES                    │
│  • Gradient descent                 │
│  • Cross entropy                    │
│  • Regularization                   │
│                                     │
│  ───────────────────────────────    │
│                                     │
│  RESULTS                            │
│  [Result items...]                  │
│                                     │
└─────────────────────────────────────┘
```

**Features:**
- Full-screen overlay on mobile
- Back button to close
- Auto-focus on open
- Virtual keyboard handling
- Recent searches (localStorage)
- Smooth open/close animation

#### 3.2 useMobileDetect Hook
```typescript
// src/hooks/useMobileDetect.ts
interface MobileDetectResult {
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
  isTouchDevice: boolean;
}

function useMobileDetect(): MobileDetectResult
```

---

### Task 4: Touch Gestures for Modal

#### 4.1 Update DetailModal for Swipe
```typescript
// Add to DetailModal.tsx
- Swipe left: Next concept
- Swipe right: Previous concept
- Swipe down: Close modal (optional)
```

**Implementation:**
- Use framer-motion's drag gesture
- Threshold: 50px horizontal for nav, 100px down for close
- Visual feedback during drag
- Snap back if threshold not met

---

### Task 5: Animation Polish

#### 5.1 Search Animations
- Results dropdown: Slide down + fade (200ms)
- Result items: Stagger fade in (50ms each)
- Highlight pulse on match
- Clear button: Scale in/out

#### 5.2 Block Animations Polish
- Smoother expand/collapse (already good)
- Stagger children appearance
- Hover lift effect refinement

#### 5.3 Reduced Motion Support
```css
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
```

---

### Task 6: Responsive Design Adjustments

#### 6.1 Header Responsiveness
- Mobile: Hide logo text, show only icon
- Mobile: Search icon button instead of input
- Tablet: Compact search input
- Desktop: Full search input

#### 6.2 Block Grid Responsiveness
Already implemented, verify:
- Mobile (<640px): 1 column
- Tablet (640-1024px): 2 columns
- Desktop (>1024px): 3 columns

#### 6.3 Modal Responsiveness
- Mobile: Full screen modal
- Tablet: 90% width modal
- Desktop: max-w-4xl modal

#### 6.4 Touch Targets
- Minimum 44x44px for all interactive elements
- Increase padding on mobile for buttons
- Larger close buttons on modal

---

## Dependencies

```bash
# Fuzzy search (optional but recommended)
npm install fuse.js

# Types for Fuse.js
npm install -D @types/fuse.js
```

**Note:** Can implement without Fuse.js using basic string matching first, add fuzzy search as enhancement.

---

## State Updates

### Add to AppState
```typescript
interface AppState {
  // ... existing

  // Search enhancements
  searchFocused: boolean;
  recentSearches: string[];  // Persisted to localStorage
}
```

### Add Actions
```typescript
type AppAction =
  | // ... existing
  | { type: 'SET_SEARCH_FOCUSED'; payload: boolean }
  | { type: 'ADD_RECENT_SEARCH'; payload: string }
  | { type: 'CLEAR_RECENT_SEARCHES' }
```

---

## File Structure After Week 3

```
src/
├── components/
│   ├── common/
│   │   ├── Badge.tsx
│   │   ├── Breadcrumb.tsx
│   │   ├── CodeBlock.tsx
│   │   ├── EmptyState.tsx           # NEW
│   │   └── index.ts
│   ├── search/                       # NEW DIRECTORY
│   │   ├── SearchInput.tsx
│   │   ├── SearchResults.tsx
│   │   ├── SearchResultItem.tsx
│   │   ├── SearchHighlight.tsx
│   │   ├── MobileSearchOverlay.tsx
│   │   └── index.ts
│   ├── blocks/
│   ├── detail/
│   └── layout/
│       ├── Header.tsx               # UPDATE - integrate search
│       └── MainContent.tsx
├── hooks/
│   ├── useConceptData.ts
│   ├── useKeyboardShortcuts.ts
│   ├── useFocusTrap.ts
│   ├── useModalNavigation.ts
│   ├── useSearch.ts                 # NEW
│   ├── useDebounce.ts               # NEW
│   ├── useMobileDetect.ts           # NEW
│   └── index.ts
├── utils/
│   ├── cn.ts
│   ├── constants.ts
│   ├── dataHelpers.ts
│   └── searchHelpers.ts             # NEW
├── context/
│   └── AppContext.tsx               # UPDATE - add search state
└── types/
    ├── concept.ts
    ├── navigation.ts
    ├── state.ts                     # UPDATE - add search types
    └── search.ts                    # NEW
```

---

## Implementation Order

| Day | Tasks | Priority |
|-----|-------|----------|
| **Day 1** | useDebounce, searchHelpers.ts, search types | Critical |
| **Day 2** | useSearch hook, SearchInput component | Critical |
| **Day 3** | SearchResults, SearchResultItem, SearchHighlight | Critical |
| **Day 4** | Header integration, keyboard shortcuts (/, Ctrl+K) | High |
| **Day 5** | MobileSearchOverlay, useMobileDetect | High |
| **Day 6** | Touch gestures for modal, animation polish | Medium |
| **Day 7** | Responsive adjustments, testing, bug fixes | High |

---

## Detailed Day-by-Day Plan

### Day 1: Search Foundation

**Files to create:**
1. `src/types/search.ts`
2. `src/hooks/useDebounce.ts`
3. `src/utils/searchHelpers.ts`

**Search types:**
```typescript
// src/types/search.ts
export interface SearchResult {
  concept: Concept;
  score: number;
  matchedField: 'name' | 'tags' | 'content';
  matchRanges: Array<{ start: number; end: number }>;
}

export interface SearchOptions {
  maxResults?: number;
  fuzzy?: boolean;
  threshold?: number;
}

export type GroupedSearchResults = Map<Category, SearchResult[]>;
```

**Search algorithm implementation:**
- Normalize and tokenize query
- Score concepts based on match location
- Sort by score
- Extract match ranges for highlighting

---

### Day 2: Search Hook & Input

**Files to create:**
1. `src/hooks/useSearch.ts`
2. `src/components/search/SearchInput.tsx`
3. `src/components/search/index.ts`

**Update:**
1. `src/context/AppContext.tsx` - Add search-focused state
2. `src/types/state.ts` - Add new state/action types

**SearchInput features:**
- Controlled input synced with useSearch
- Clear button
- Loading state
- Focus/blur handling
- Keyboard event handling

---

### Day 3: Search Results

**Files to create:**
1. `src/components/search/SearchResults.tsx`
2. `src/components/search/SearchResultItem.tsx`
3. `src/components/search/SearchHighlight.tsx`
4. `src/components/common/EmptyState.tsx`

**SearchResults features:**
- Category grouping
- Keyboard navigation (up/down)
- Selected item highlighting
- Scroll selected into view
- Empty state handling
- Results count

---

### Day 4: Header Integration

**Update:**
1. `src/components/layout/Header.tsx`
2. `src/hooks/useKeyboardShortcuts.ts`

**Changes:**
- Replace disabled input with SearchInput
- Add SearchResults dropdown
- Position dropdown correctly
- Click outside to close
- Add global keyboard shortcuts (/, Ctrl+K)
- Handle search result selection

---

### Day 5: Mobile Search

**Files to create:**
1. `src/components/search/MobileSearchOverlay.tsx`
2. `src/hooks/useMobileDetect.ts`

**MobileSearchOverlay features:**
- Full-screen overlay
- AnimatePresence for smooth transitions
- Auto-focus input
- Recent searches list
- Results display
- Close on result selection

**Header mobile update:**
- Show search icon button on mobile
- Open MobileSearchOverlay on tap

---

### Day 6: Touch & Polish

**Update:**
1. `src/components/detail/DetailModal.tsx` - Add swipe gestures
2. `src/index.css` - Animation refinements

**Swipe implementation:**
```tsx
<motion.div
  drag="x"
  dragConstraints={{ left: 0, right: 0 }}
  onDragEnd={(e, info) => {
    if (info.offset.x > 50) goToPrevious();
    if (info.offset.x < -50) goToNext();
  }}
>
```

**Animation polish:**
- Refine transition timings
- Add subtle micro-interactions
- Ensure reduced motion support

---

### Day 7: Responsive & Testing

**Tasks:**
1. Test on various screen sizes
2. Test keyboard navigation
3. Test touch interactions
4. Fix any responsive issues
5. Cross-browser testing
6. Performance check
7. Accessibility audit

**Responsive fixes:**
- Ensure 44px touch targets
- Check modal sizing on all devices
- Verify search dropdown positioning
- Test with virtual keyboard on mobile

---

## Testing Checklist

### Search Tests
- [ ] Search updates as user types (with debounce)
- [ ] Results match by name, tags, content
- [ ] Results grouped by category
- [ ] Matching text highlighted
- [ ] Clicking result opens modal
- [ ] Clicking result expands ancestors
- [ ] Empty state shown for no results
- [ ] Clear button works
- [ ] Search clears on Escape

### Keyboard Tests
- [ ] `/` focuses search (when not in input)
- [ ] `Ctrl+K` / `Cmd+K` focuses search
- [ ] `↑` `↓` navigate results
- [ ] `Enter` selects highlighted result
- [ ] `Escape` clears search or closes dropdown

### Mobile Tests
- [ ] Search icon visible on mobile header
- [ ] Tap opens full-screen search
- [ ] Virtual keyboard doesn't obscure input
- [ ] Results scrollable on mobile
- [ ] Touch targets are 44px minimum
- [ ] Swipe navigates modal concepts

### Responsive Tests
- [ ] Header adapts to screen size
- [ ] Search dropdown positions correctly
- [ ] Modal sizing appropriate per device
- [ ] Block grid columns adjust
- [ ] No horizontal scroll

### Accessibility Tests
- [ ] Search input has label
- [ ] Results have proper ARIA roles
- [ ] Selected result announced
- [ ] Focus visible on all elements
- [ ] Reduced motion respected

---

## Success Criteria

Week 3 is complete when:

1. ✅ Search input accepts queries and shows results in real-time
2. ✅ Results are grouped by category with highlighted matches
3. ✅ Keyboard navigation works (/, Ctrl+K, arrows, Enter, Escape)
4. ✅ Clicking result opens concept modal and expands path
5. ✅ Mobile has full-screen search overlay
6. ✅ Touch gestures work for modal navigation
7. ✅ All animations are smooth and polished
8. ✅ Application is responsive on all screen sizes
9. ✅ Accessibility requirements met
10. ✅ No console errors or warnings

---

## Performance Considerations

### Search Performance
- Debounce prevents excessive re-renders
- Memoize search results
- Limit results to MAX_SEARCH_RESULTS (20)
- Consider Web Worker for large datasets (Week 4)

### Bundle Size
- Fuse.js adds ~15KB gzipped
- Can use lighter alternative or custom implementation
- Code split mobile overlay component

---

## Notes

- Theme toggle already implemented in Week 1
- Loading/error states already implemented in Week 1
- Constants (debounce, max results) already defined
- State structure already has search fields
- Focus on getting basic search working first, then enhance

**Dependencies on Week 2:**
- Modal component for result selection ✓
- Keyboard shortcuts hook ✓
- Focus trap for mobile overlay ✓

**Optional Enhancements:**
- Fuzzy search with Fuse.js
- Search history persistence
- Search analytics
- Voice search (future)
