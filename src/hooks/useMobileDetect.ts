/**
 * Hook for detecting mobile/tablet/desktop devices
 */
import { useState, useEffect } from 'react';

export interface MobileDetectResult {
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
  isTouchDevice: boolean;
}

// Breakpoints (matching Tailwind defaults)
const MOBILE_BREAKPOINT = 640; // sm
const TABLET_BREAKPOINT = 1024; // lg

/**
 * Detect if the current device is a touch device
 */
function detectTouchDevice(): boolean {
  if (typeof window === 'undefined') return false;

  return (
    'ontouchstart' in window ||
    navigator.maxTouchPoints > 0 ||
    // @ts-expect-error - msMaxTouchPoints is IE-specific
    navigator.msMaxTouchPoints > 0
  );
}

/**
 * Get device type based on window width
 */
function getDeviceType(width: number): Pick<MobileDetectResult, 'isMobile' | 'isTablet' | 'isDesktop'> {
  return {
    isMobile: width < MOBILE_BREAKPOINT,
    isTablet: width >= MOBILE_BREAKPOINT && width < TABLET_BREAKPOINT,
    isDesktop: width >= TABLET_BREAKPOINT,
  };
}

/**
 * Custom hook for detecting device type
 */
export function useMobileDetect(): MobileDetectResult {
  const [state, setState] = useState<MobileDetectResult>(() => {
    // Initial state (SSR-safe)
    if (typeof window === 'undefined') {
      return {
        isMobile: false,
        isTablet: false,
        isDesktop: true,
        isTouchDevice: false,
      };
    }

    return {
      ...getDeviceType(window.innerWidth),
      isTouchDevice: detectTouchDevice(),
    };
  });

  useEffect(() => {
    // Update touch device detection on mount
    setState((prev) => ({
      ...prev,
      isTouchDevice: detectTouchDevice(),
    }));

    // Handle resize
    const handleResize = () => {
      setState((prev) => ({
        ...prev,
        ...getDeviceType(window.innerWidth),
      }));
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return state;
}

export default useMobileDetect;
