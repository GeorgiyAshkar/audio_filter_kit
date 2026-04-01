import Fastify from 'fastify';
import cors from '@fastify/cors';
import { z } from 'zod';
import { runPipeline } from './filters.js';
import { calcMetrics } from './metrics.js';

const app = Fastify({ logger: true });
await app.register(cors, { origin: true });

const requestSchema = z.object({
  signal: z.array(z.number()),
  sampleRate: z.number().int().positive(),
  pipeline: z.array(
    z.object({
      id: z.string(),
      type: z.enum(['gain', 'normalize', 'moving_average', 'high_pass', 'low_pass']),
      name: z.string(),
      params: z.record(z.number())
    })
  )
});

app.get('/api/health', async () => ({ ok: true }));

app.post('/api/process', async (req, reply) => {
  const parsed = requestSchema.safeParse(req.body);
  if (!parsed.success) return reply.code(400).send({ error: parsed.error.flatten() });

  const { signal, sampleRate, pipeline } = parsed.data;
  const raw = Float32Array.from(signal);
  const t0 = performance.now();
  const processed = runPipeline(raw, pipeline);
  const latencyMs = performance.now() - t0;
  const metrics = calcMetrics(raw, processed, latencyMs);

  return { processed: Array.from(processed), sampleRate, metrics };
});

app.listen({ port: 8080, host: '0.0.0.0' });
