/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Category colors
        category: {
          foundations: {
            light: '#3b82f6',
            dark: '#60a5fa',
            bg: '#eff6ff',
            'bg-dark': '#1e3a5f',
          },
          data: {
            light: '#22c55e',
            dark: '#4ade80',
            bg: '#f0fdf4',
            'bg-dark': '#14532d',
          },
          learning: {
            light: '#f97316',
            dark: '#fb923c',
            bg: '#fff7ed',
            'bg-dark': '#7c2d12',
          },
          optimization: {
            light: '#eab308',
            dark: '#facc15',
            bg: '#fefce8',
            'bg-dark': '#713f12',
          },
          regularization: {
            light: '#a855f7',
            dark: '#c084fc',
            bg: '#faf5ff',
            'bg-dark': '#581c87',
          },
          evaluation: {
            light: '#14b8a6',
            dark: '#2dd4bf',
            bg: '#f0fdfa',
            'bg-dark': '#134e4a',
          },
          supervised: {
            light: '#6366f1',
            dark: '#818cf8',
            bg: '#eef2ff',
            'bg-dark': '#3730a3',
          },
          unsupervised: {
            light: '#ec4899',
            dark: '#f472b6',
            bg: '#fdf2f8',
            'bg-dark': '#831843',
          },
          neural: {
            light: '#ef4444',
            dark: '#f87171',
            bg: '#fef2f2',
            'bg-dark': '#7f1d1d',
          },
          mlops: {
            light: '#64748b',
            dark: '#94a3b8',
            bg: '#f8fafc',
            'bg-dark': '#334155',
          },
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      animation: {
        'expand': 'expand 0.3s ease-out',
        'collapse': 'collapse 0.2s ease-in',
        'fade-in': 'fadeIn 0.15s ease',
        'scale-in': 'scaleIn 0.2s ease',
        'slide-down': 'slideDown 0.3s ease-out',
      },
      keyframes: {
        expand: {
          '0%': { opacity: '0', transform: 'scaleY(0.8)' },
          '100%': { opacity: '1', transform: 'scaleY(1)' },
        },
        collapse: {
          '0%': { opacity: '1', transform: 'scaleY(1)' },
          '100%': { opacity: '0', transform: 'scaleY(0.8)' },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        scaleIn: {
          '0%': { opacity: '0', transform: 'scale(0.95)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        slideDown: {
          '0%': { opacity: '0', transform: 'translateY(-10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        '128': '32rem',
      },
      maxWidth: {
        '8xl': '88rem',
        '9xl': '96rem',
      },
    },
  },
  plugins: [],
}
