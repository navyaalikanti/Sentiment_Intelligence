/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Manrope', 'ui-sans-serif', 'system-ui'],
      },
      colors: {
        bg: {
          950: '#050816',
          900: '#09111f',
          800: '#111827',
        },
        panel: {
          900: 'rgba(10, 16, 31, 0.72)',
          800: 'rgba(15, 23, 42, 0.78)',
        },
        accent: {
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
        },
      },
      boxShadow: {
        glow: '0 0 0 1px rgba(125, 211, 252, 0.12), 0 24px 80px rgba(2, 132, 199, 0.18)',
      },
      backgroundImage: {
        'dashboard-radial':
          'radial-gradient(circle at top left, rgba(14, 165, 233, 0.22), transparent 30%), radial-gradient(circle at top right, rgba(129, 140, 248, 0.18), transparent 25%), linear-gradient(180deg, rgba(5, 8, 22, 0.96), rgba(9, 17, 31, 1))',
      },
    },
  },
  plugins: [],
};
