/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        display: ['Orbitron', 'sans-serif'],
        body: ['"Plus Jakarta Sans"', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      colors: {
        space: {
          950: '#050914',
          900: '#0a0e1a',
          800: '#111827',
          700: '#1e293b',
          600: '#334155',
        },
        lunar: {
          cyan: '#06b6d4',
          teal: '#2dd4bf',
          amber: '#f59e0b',
          emerald: '#10b981',
          rose: '#f43f5e',
        },
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-out forwards',
        'slide-up': 'slideUp 0.4s ease-out forwards',
        'pulse-glow': 'pulseGlow 2s ease-in-out infinite',
        'drift': 'drift 20s linear infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        pulseGlow: {
          '0%, 100%': { boxShadow: '0 0 20px rgba(6,182,212,0.15)' },
          '50%': { boxShadow: '0 0 40px rgba(6,182,212,0.3)' },
        },
        drift: {
          '0%': { transform: 'translateY(0) translateX(0)' },
          '25%': { transform: 'translateY(-2px) translateX(1px)' },
          '50%': { transform: 'translateY(0) translateX(0)' },
          '75%': { transform: 'translateY(2px) translateX(-1px)' },
          '100%': { transform: 'translateY(0) translateX(0)' },
        },
      },
    },
  },
  plugins: [],
}
