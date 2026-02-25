
/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                'brand-dark': '#0f172a',
                'brand-blue': '#3b82f6',
                'brand-accent': '#22c55e',
            }
        },
    },
    plugins: [],
    darkMode: 'class',
}
