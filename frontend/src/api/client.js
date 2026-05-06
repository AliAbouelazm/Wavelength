const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

async function request(method, path, body) {
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json' },
  };
  if (body !== undefined) {
    opts.body = JSON.stringify(body);
  }
  const resp = await fetch(`${BASE_URL}${path}`, opts);
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || resp.statusText);
  }
  return resp.json();
}

export const fetchMoods = () => request('GET', '/moods');
export const recommend = (body) => request('POST', '/recommend', body);
export const analyze = (body) => request('POST', '/analyze', body);
export const fetchExplore = () => request('GET', '/explore');
export const fetchModelMetrics = () => request('GET', '/model/metrics');
