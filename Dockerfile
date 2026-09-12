# The served endpoint. One artifact in, retrieval out — no database, no cache
# directory, no keys.
#
# `pnpm serve:prepare` reduces cache/ (300MB of vectors for seven chunking
# configurations) to one 4.6MB file holding the configuration the README
# describes. That file is the only project data this image carries, which is
# what makes the build reproducible from a checkout plus one artifact rather
# than from a warm cache nobody else has.
FROM node:24-slim AS base
# Two statements, not one: $PNPM_HOME is not yet set while the same ENV is
# being evaluated, and Docker warns about exactly that.
ENV PNPM_HOME=/pnpm
ENV PATH=$PNPM_HOME:$PATH
RUN corepack enable

WORKDIR /app

# Manifests first, so a code change does not re-resolve the dependency tree.
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY packages/chunking/package.json  packages/chunking/
COPY packages/embed/package.json     packages/embed/
COPY packages/ingest/package.json    packages/ingest/
COPY packages/retrieve/package.json  packages/retrieve/
COPY packages/evals/package.json     packages/evals/
COPY packages/report/package.json    packages/report/
COPY packages/serve/package.json     packages/serve/
# --prod: the server needs no tsc and no vitest. Nothing is compiled here —
# Node strips the types at run time — so the dev tree is weight with no use.
RUN pnpm install --frozen-lockfile --prod

# The project runs TypeScript directly through Node's type stripping, so there
# is no build step and no dist/ — the source is what runs, here as everywhere.
COPY packages/ packages/
COPY configs.yaml corpus.yaml ./

# The prepared index. Not in git: it is derived from cache/, and 4.6MB of
# vectors do not belong in a repository. Build it with `pnpm serve:prepare`.
COPY serve-index.json ./

# The embedding model, ~90MB, downloaded once at build time rather than on the
# first request — a cold container should not make its first user wait for a
# model download, and an image that needs the network to answer is not
# deployable anywhere offline.
ENV TRANSFORMERS_CACHE=/app/.model-cache
RUN node --experimental-strip-types -e \
  'import("./packages/embed/src/local.ts").then((m) => m.embedLocal(["warm"]))'

ENV NODE_ENV=production PORT=8080 INDEX=/app/serve-index.json
EXPOSE 8080

# Unprivileged: the process reads one file and answers questions, and needs
# nothing else on the filesystem.
USER node

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD node -e "fetch('http://localhost:'+process.env.PORT+'/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

CMD ["node", "--experimental-strip-types", "packages/serve/src/server.ts"]
