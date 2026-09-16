/**
 * Victor Chat WebView - Main Entry Point
 */

import './styles/theme.css';
import { mount } from 'svelte';
import App from './components/App.svelte';

const app = mount(App, {
  target: document.getElementById('app')!,
});

export default app;
