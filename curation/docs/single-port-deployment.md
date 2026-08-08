# Single-port deployment

Build the frontend with `npm run build`, then start the backend. When `curation/frontend/dist` exists, FastAPI serves the built frontend and API from the same backend port, so annotators need only one service endpoint.
