import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  server: { host: true, port: 5173 },
  build: {
    target: 'es2020',
    minify: 'terser',
    terserOptions: { compress: { passes: 2 }, format: { comments: false } },
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules/three')) return 'three';
          return undefined;
        },
      },
    },
    chunkSizeWarningLimit: 1600,
  },
});
