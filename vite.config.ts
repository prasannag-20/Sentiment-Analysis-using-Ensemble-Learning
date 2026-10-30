import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {defineConfig} from 'vite';

export default defineConfig(() => {
  // If running in AI Studio container, avoid launching local browser window.
  // When running locally in VS Code (where DISABLE_HMR is not set), open browser automatically.
  const isAiStudio = process.env.DISABLE_HMR === 'true';

  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      // Automatically open application in browser when running "npm run dev"
      open: isAiStudio ? false : (process.env.BROWSER || true),
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modify—file watching is disabled to prevent flickering during agent edits.
      hmr: !isAiStudio,
      // Disable file watching when DISABLE_HMR is true to save CPU during agent edits.
      watch: isAiStudio ? null : {},
    },
  };
});
