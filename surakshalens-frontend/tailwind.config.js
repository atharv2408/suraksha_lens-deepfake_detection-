/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx}",
  ],
  theme: {
    extend: {
      colors: {
        neonPurple: "#9b5cff",
        deepBlack: "#0a0a0f",
        cardDark: "#14141c",
      },
      boxShadow: {
        neon: "0 0 15px #9b5cff",
      },
    },
  },
  plugins: [],
};
