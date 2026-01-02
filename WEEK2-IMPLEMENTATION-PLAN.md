# Week 2: Navigation & Modal - Implementation Plan

## Overview

**Goal**: Build the Detail Modal with tabbed content, breadcrumb navigation, keyboard shortcuts, and code syntax highlighting. This week transforms the app from a browsable hierarchy into a full learning tool.

**Deliverable**: Full navigation working with modal details view displaying Technical/Simple/Example tabs.

---

## Week 2 Tasks from Project Plan

| # | Task | Description |
|---|------|-------------|
| 1 | Expand/collapse animation polish | Improve existing animations |
| 2 | Build DetailModal component | Full-screen modal with concept details |
| 3 | Create tabbed content view | Technical / Simple / Example tabs |
| 4 | Add breadcrumb navigation | Show path to current concept |
| 5 | Implement keyboard shortcuts | Navigation without mouse |
| 6 | Code syntax highlighting | Highlight Python/JS code in examples |

---

## Detailed Requirements Analysis

### 2.1 DetailModal Component

**From Project Plan (Section 3.3):**

```
┌─────────────────────────────────────────────────────┐
│  [X]                                                │
│  ┌─────────────────────────────────────────────┐   │
│  │         CONCEPT NAME                         │   │
│  │         Section 4.2.1 | Intermediate         │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  ┌──────────┬──────────┬──────────┐               │
│  │ Technical│  Simple  │  Example │               │
│  └──────────┴──────────┴──────────┘               │
│  ┌─────────────────────────────────────────────┐   │
│  │                                             │   │
│  │  [Tab Content Area]                         │   │
│  │                                             │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  Related: [Concept A] [Concept B] [Concept C]      │
│  Prerequisites: [Concept X] [Concept Y]             │
│                                                     │
│  ┌────────────────┐  ┌────────────────┐           │
│  │  ◄ Previous    │  │    Next ►      │           │
│  └────────────────┘  └────────────────┘           │
└─────────────────────────────────────────────────────┘
```

**Modal Behavior:**
- Open animation: Fade in + scale up (150ms)
- Close: Click outside, X button, Escape key
- Navigation: Previous/Next within same parent category
- Deep linking: URL updates to include concept ID (optional)
- Keyboard: Arrow keys for prev/next, Tab for tabs

**Tab Contents:**
| Tab | Content |
|-----|---------|
| Technical | Full, detailed explanation with technical terms |
| Simple | Plain English explanation, analogies, no jargon |
| Example | Code snippets, tables, visuals, real-world scenarios |

### 2.2 Breadcrumb Navigation

**From Project Plan (Section 3.5):**

```
Home > Optimization > Gradient-Based > SGD
```

- Each segment is clickable
- Mobile: Collapsed to show only current + parent
- Highlights current depth

### 2.3 Keyboard Shortcuts

**From Project Plan (Section 5.2):**

| Shortcut | Action |
|----------|--------|
| `Escape` | Close modal |
| `←` `→` | Previous/Next concept in modal |
| `1` `2` `3` | Switch tabs in modal |
| `Space` or `Enter` | Toggle expand (when block focused) |

### 2.4 Accessibility Requirements

**From Project Plan (Section 9):**

```tsx
// Modal ARIA attributes
<div
  role="dialog"
  aria-modal="true"
  aria-labelledby="modal-title"
  aria-describedby="modal-description"
>
```

- Modal traps focus when open
- Returns focus to trigger element on close
- Visible focus indicators

---

## New Files to Create

```
src/
├── components/
│   ├── common/
│   │   ├── Breadcrumb.tsx          # Breadcrumb navigation
│   │   ├── Badge.tsx               # Reusable badge component
│   │   ├── CodeBlock.tsx           # Syntax highlighted code
│   │   └── index.ts
│   └── detail/
│       ├── DetailModal.tsx         # Modal wrapper with overlay
│       ├── DetailHeader.tsx        # Concept name, badges, close button
│       ├── DetailTabs.tsx          # Tab navigation component
│       ├── TechnicalTab.tsx        # Technical explanation content
│       ├── SimpleTab.tsx           # Simple explanation content
│       ├── ExampleTab.tsx          # Example with code/tables
│       ├── RelatedConcepts.tsx     # Related & prerequisite links
│       ├── ModalNavigation.tsx     # Previous/Next buttons
│       └── index.ts
├── hooks/
│   ├── useKeyboardShortcuts.ts     # Global keyboard shortcuts
│   ├── useFocusTrap.ts             # Focus trapping for modal
│   └── useModalNavigation.ts       # Prev/Next concept logic
└── utils/
    └── codeHighlight.ts            # Code syntax highlighting setup
```

---

## Implementation Tasks Breakdown

### Task 1: Common Components (Foundation)

#### 1.1 Badge Component
```typescript
// src/components/common/Badge.tsx
interface BadgeProps {
  variant: 'section' | 'difficulty' | 'category' | 'tag';
  children: React.ReactNode;
  color?: string;
}
```
- Reusable badge for section IDs, difficulty levels, tags
- Supports different color schemes per variant

#### 1.2 CodeBlock Component
```typescript
// src/components/common/CodeBlock.tsx
interface CodeBlockProps {
  code: string;
  language: 'python' | 'javascript' | 'typescript' | 'pseudocode';
  showLineNumbers?: boolean;
}
```
- Syntax highlighting using `highlight.js` or `prism-react-renderer`
- Copy to clipboard button
- Line numbers (optional)
- Dark/light theme support

#### 1.3 Breadcrumb Component
```typescript
// src/components/common/Breadcrumb.tsx
interface BreadcrumbProps {
  conceptId: string;
  onNavigate: (conceptId: string) => void;
}
```
- Shows path: Home > Category > Parent > Current
- Each segment clickable (expands to that level)
- Mobile: collapses middle segments

---

### Task 2: Detail Modal Components

#### 2.1 DetailModal (Main Wrapper)
```typescript
// src/components/detail/DetailModal.tsx
interface DetailModalProps {
  conceptId: string;
  isOpen: boolean;
  onClose: () => void;
}
```

**Features:**
- Framer Motion animations (fade + scale)
- Click outside to close
- Escape key to close
- Focus trap inside modal
- Scroll lock on body when open
- Returns focus to trigger on close

**Structure:**
```tsx
<AnimatePresence>
  {isOpen && (
    <>
      <motion.div className="overlay" onClick={onClose} />
      <motion.div className="modal-content" role="dialog" aria-modal="true">
        <DetailHeader />
        <DetailTabs />
        <TabContent />
        <RelatedConcepts />
        <ModalNavigation />
      </motion.div>
    </>
  )}
</AnimatePresence>
```

#### 2.2 DetailHeader
```typescript
// src/components/detail/DetailHeader.tsx
interface DetailHeaderProps {
  concept: Concept;
  onClose: () => void;
}
```

**Contains:**
- Close button (X) - top right
- Concept name (h2)
- Section ID badge
- Difficulty badge
- Category indicator

#### 2.3 DetailTabs
```typescript
// src/components/detail/DetailTabs.tsx
interface DetailTabsProps {
  activeTab: 'technical' | 'simple' | 'example';
  onTabChange: (tab: 'technical' | 'simple' | 'example') => void;
}
```

**Features:**
- Three tabs: Technical, Simple, Example
- Active tab highlighted
- Keyboard accessible (arrow keys between tabs)
- Animated underline indicator

#### 2.4 Tab Content Components

**TechnicalTab.tsx:**
- Renders `fullExplanation` as formatted text
- Supports markdown-like formatting (paragraphs, lists)
- Preserves whitespace for formulas

**SimpleTab.tsx:**
- Renders `simpleExplanation`
- More casual styling
- Emphasis on clarity

**ExampleTab.tsx:**
- Renders `example.description`
- Renders `example.code` with CodeBlock (if present)
- Renders `example.table` as styled table (if present)
- Renders `example.visual` as image (if present)

#### 2.5 RelatedConcepts
```typescript
// src/components/detail/RelatedConcepts.tsx
interface RelatedConceptsProps {
  relatedIds: string[];
  prerequisiteIds: string[];
  onNavigate: (conceptId: string) => void;
}
```

**Features:**
- Two sections: "Related Concepts" and "Prerequisites"
- Each concept as clickable chip/badge
- Click opens that concept in modal
- Shows concept name (resolved from ID)

#### 2.6 ModalNavigation
```typescript
// src/components/detail/ModalNavigation.tsx
interface ModalNavigationProps {
  currentConceptId: string;
  onPrevious: () => void;
  onNext: () => void;
  hasPrevious: boolean;
  hasNext: boolean;
}
```

**Features:**
- Previous/Next buttons
- Disabled state when at boundaries
- Shows sibling concept names as hints
- Keyboard: Left/Right arrow keys

---

### Task 3: Custom Hooks

#### 3.1 useKeyboardShortcuts
```typescript
// src/hooks/useKeyboardShortcuts.ts
interface ShortcutConfig {
  key: string;
  ctrlKey?: boolean;
  shiftKey?: boolean;
  action: () => void;
  enabled?: boolean;
}

function useKeyboardShortcuts(shortcuts: ShortcutConfig[]): void
```

**Shortcuts to implement:**
- `Escape` - Close modal
- `ArrowLeft` - Previous concept (when modal open)
- `ArrowRight` - Next concept (when modal open)
- `1` - Switch to Technical tab
- `2` - Switch to Simple tab
- `3` - Switch to Example tab

#### 3.2 useFocusTrap
```typescript
// src/hooks/useFocusTrap.ts
function useFocusTrap(
  containerRef: RefObject<HTMLElement>,
  isActive: boolean
): void
```

**Features:**
- Trap focus within modal when open
- Tab cycles through focusable elements
- Shift+Tab cycles backwards
- First focusable element focused on open

#### 3.3 useModalNavigation
```typescript
// src/hooks/useModalNavigation.ts
interface ModalNavigationResult {
  previousConcept: Concept | null;
  nextConcept: Concept | null;
  goToPrevious: () => void;
  goToNext: () => void;
}

function useModalNavigation(currentConceptId: string): ModalNavigationResult
```

**Logic:**
- Find siblings (concepts with same parentId)
- Sort by sectionId
- Determine previous/next in sequence

---

### Task 4: Code Syntax Highlighting

**Option A: highlight.js** (lighter)
```bash
npm install highlight.js
```

**Option B: prism-react-renderer** (React-native)
```bash
npm install prism-react-renderer
```

**Recommendation:** Use `prism-react-renderer` for better React integration.

**Implementation:**
```typescript
// src/utils/codeHighlight.ts
import { Highlight, themes } from 'prism-react-renderer';

// Theme selection based on app theme
export const getCodeTheme = (isDark: boolean) =>
  isDark ? themes.vsDark : themes.vsLight;
```

---

### Task 5: State Updates

**Add to AppContext:**
```typescript
// New action types
| { type: 'OPEN_MODAL'; payload: string }
| { type: 'CLOSE_MODAL' }
| { type: 'NAVIGATE_MODAL'; payload: 'previous' | 'next' }
```

**Update existing behavior:**
- `SELECT_CONCEPT` already opens modal (keep as is)
- Add tracking of trigger element for focus return

---

### Task 6: Integration with Existing Components

**Update Block.tsx:**
- Info icon click triggers `SELECT_CONCEPT` action
- Already implemented in Week 1 ✓

**Update App.tsx:**
- Add DetailModal component (conditionally rendered)
- Connect to AppContext state

**Update Header.tsx:**
- Add Breadcrumb component
- Show when a concept is selected or path is deep

---

## Dependency Installation

```bash
npm install prism-react-renderer
```

No other new dependencies needed - Framer Motion already installed.

---

## File Structure After Week 2

```
src/
├── components/
│   ├── common/
│   │   ├── Badge.tsx               # NEW
│   │   ├── Breadcrumb.tsx          # NEW
│   │   ├── CodeBlock.tsx           # NEW
│   │   └── index.ts                # NEW
│   ├── detail/
│   │   ├── DetailModal.tsx         # NEW
│   │   ├── DetailHeader.tsx        # NEW
│   │   ├── DetailTabs.tsx          # NEW
│   │   ├── TechnicalTab.tsx        # NEW
│   │   ├── SimpleTab.tsx           # NEW
│   │   ├── ExampleTab.tsx          # NEW
│   │   ├── RelatedConcepts.tsx     # NEW
│   │   ├── ModalNavigation.tsx     # NEW
│   │   └── index.ts                # NEW
│   ├── blocks/                      # Existing (minor updates)
│   └── layout/                      # Existing (add Breadcrumb)
├── hooks/
│   ├── useConceptData.ts           # Existing
│   ├── useKeyboardShortcuts.ts     # NEW
│   ├── useFocusTrap.ts             # NEW
│   ├── useModalNavigation.ts       # NEW
│   └── index.ts                    # UPDATE
├── context/
│   └── AppContext.tsx              # UPDATE (focus return)
└── App.tsx                         # UPDATE (add modal)
```

---

## Implementation Order

| Day | Tasks | Priority |
|-----|-------|----------|
| **Day 1** | Common components (Badge, CodeBlock) | High |
| **Day 2** | Breadcrumb component + integration | High |
| **Day 3** | DetailModal shell + DetailHeader + DetailTabs | Critical |
| **Day 4** | Tab content components (Technical, Simple, Example) | Critical |
| **Day 5** | RelatedConcepts + ModalNavigation | High |
| **Day 6** | Keyboard shortcuts + Focus trap hooks | High |
| **Day 7** | Integration, polish, testing | High |

---

## Detailed Day-by-Day Plan

### Day 1: Common Components

**Files to create:**
1. `src/components/common/Badge.tsx`
2. `src/components/common/CodeBlock.tsx`
3. `src/components/common/index.ts`

**Install:**
```bash
npm install prism-react-renderer
```

**Badge variants:**
- `section` - Blue background, shows section ID like "1.2.3"
- `difficulty` - Green/Yellow/Red based on level
- `category` - Category color from constants
- `tag` - Gray, for search tags

**CodeBlock features:**
- Python syntax highlighting
- JavaScript/TypeScript highlighting
- Copy button
- Line numbers
- Responsive (horizontal scroll on mobile)

---

### Day 2: Breadcrumb Component

**Files to create:**
1. `src/components/common/Breadcrumb.tsx`

**Update:**
1. `src/components/layout/Header.tsx` - Add breadcrumb
2. `src/hooks/useConceptData.ts` - Ensure `useConceptPath` works

**Breadcrumb behavior:**
- Home icon + "Home" for root
- Category name for category level
- Parent concept names
- Current concept (non-clickable, highlighted)
- Mobile: Show "..." for middle items

---

### Day 3: Modal Shell

**Files to create:**
1. `src/components/detail/DetailModal.tsx`
2. `src/components/detail/DetailHeader.tsx`
3. `src/components/detail/DetailTabs.tsx`
4. `src/components/detail/index.ts`

**DetailModal structure:**
```tsx
<AnimatePresence>
  {isOpen && (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Overlay */}
      <motion.div
        className="absolute inset-0 bg-black/50"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
      />

      {/* Modal Content */}
      <motion.div
        className="relative bg-white dark:bg-gray-800 rounded-2xl
                   max-w-4xl w-full mx-4 max-h-[90vh] overflow-hidden"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        role="dialog"
        aria-modal="true"
        aria-labelledby="modal-title"
      >
        <DetailHeader concept={concept} onClose={onClose} />
        <DetailTabs activeTab={activeTab} onTabChange={setActiveTab} />
        {/* Tab content area */}
      </motion.div>
    </div>
  )}
</AnimatePresence>
```

---

### Day 4: Tab Content Components

**Files to create:**
1. `src/components/detail/TechnicalTab.tsx`
2. `src/components/detail/SimpleTab.tsx`
3. `src/components/detail/ExampleTab.tsx`

**TechnicalTab:**
- Render `fullExplanation` text
- Preserve paragraph breaks
- Style for readability (line height, margins)

**SimpleTab:**
- Render `simpleExplanation` text
- Lighter, more approachable styling
- Larger font size

**ExampleTab:**
- Render `example.description`
- Conditionally render CodeBlock if `example.code` exists
- Conditionally render table if `example.table` exists
- Table component with headers and rows

---

### Day 5: Related Concepts & Navigation

**Files to create:**
1. `src/components/detail/RelatedConcepts.tsx`
2. `src/components/detail/ModalNavigation.tsx`
3. `src/hooks/useModalNavigation.ts`

**RelatedConcepts:**
- Map `relatedConcepts` IDs to concept names
- Render as clickable chips
- Same for `prerequisites`
- Click navigates modal to that concept

**ModalNavigation:**
- Previous/Next buttons at bottom
- Uses `useModalNavigation` hook
- Shows sibling concept name as label

---

### Day 6: Keyboard & Focus

**Files to create:**
1. `src/hooks/useKeyboardShortcuts.ts`
2. `src/hooks/useFocusTrap.ts`

**useKeyboardShortcuts:**
```typescript
useKeyboardShortcuts([
  { key: 'Escape', action: closeModal, enabled: isModalOpen },
  { key: 'ArrowLeft', action: goToPrevious, enabled: isModalOpen },
  { key: 'ArrowRight', action: goToNext, enabled: isModalOpen },
  { key: '1', action: () => setTab('technical'), enabled: isModalOpen },
  { key: '2', action: () => setTab('simple'), enabled: isModalOpen },
  { key: '3', action: () => setTab('example'), enabled: isModalOpen },
]);
```

**useFocusTrap:**
- Query all focusable elements in modal
- Handle Tab and Shift+Tab
- Wrap focus at boundaries

---

### Day 7: Integration & Polish

**Tasks:**
1. Update `App.tsx` to render DetailModal
2. Test all keyboard shortcuts
3. Test focus management
4. Test animations
5. Mobile responsiveness
6. Cross-browser testing
7. Fix any bugs

---

## Testing Checklist

### Modal Tests
- [ ] Opens when info icon clicked
- [ ] Closes on X button click
- [ ] Closes on overlay click
- [ ] Closes on Escape key
- [ ] Focus trapped inside modal
- [ ] Focus returns to trigger on close
- [ ] Scroll locked when open

### Tab Tests
- [ ] Technical tab displays fullExplanation
- [ ] Simple tab displays simpleExplanation
- [ ] Example tab displays code with highlighting
- [ ] Example tab displays tables correctly
- [ ] Tab switching animated
- [ ] Keyboard 1/2/3 switches tabs

### Navigation Tests
- [ ] Breadcrumb shows correct path
- [ ] Breadcrumb items clickable
- [ ] Previous/Next buttons work
- [ ] Arrow keys navigate concepts
- [ ] Related concepts clickable
- [ ] Prerequisites clickable

### Code Highlight Tests
- [ ] Python syntax highlighted
- [ ] JavaScript syntax highlighted
- [ ] Copy button works
- [ ] Dark/light theme respected
- [ ] Long code scrolls horizontally

### Accessibility Tests
- [ ] Modal has role="dialog"
- [ ] Modal has aria-modal="true"
- [ ] Tab navigation works
- [ ] Focus visible on all elements
- [ ] Screen reader announces modal

---

## Success Criteria

Week 2 is complete when:

1. ✅ Clicking info icon opens full-screen modal
2. ✅ Modal displays concept name, section, difficulty
3. ✅ Three tabs work: Technical, Simple, Example
4. ✅ Code examples have syntax highlighting
5. ✅ Related concepts are clickable links
6. ✅ Previous/Next navigation works
7. ✅ Keyboard shortcuts work (Esc, arrows, 1/2/3)
8. ✅ Breadcrumb shows navigation path
9. ✅ Focus is trapped in modal
10. ✅ Animations are smooth and polished

---

## Notes

- Expand/collapse animation already exists from Week 1
- Dark mode already implemented in Week 1
- State management foundation already in place
- Just need to extend, not rewrite

**Dependencies on Week 1:**
- AppContext with `selectedConceptId` ✓
- `useConceptPath` hook ✓
- Block component with info button ✓
- Category colors in constants ✓
