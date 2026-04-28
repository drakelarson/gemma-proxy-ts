import { Hono } from 'hono'
import { cors } from 'hono/cors'

const BASE_URL = 'https://generativelanguage.googleapis.com/v1beta'

const API_KEY = process.env.GEMINI_API_KEY

if (!API_KEY) {
  console.error('[GEMMA-PROXY] ERROR: No API key! Set GEMINI_API_KEY env var.')
}

const app = new Hono()

app.use('*', cors())

app.get('/v1/models', (c) => {
  const models = [
    { id: 'gemma-4-31b-it', object: 'model', created: 1776766009709, owned_by: 'google', gemini_model: 'gemma-4-31b-it' },
    { id: 'gemma-4-26b-a4b-it', object: 'model', created: 1776766009709, owned_by: 'google', gemini_model: 'gemma-4-26b-a4b-it' },
    { id: 'gemma-3-27b-it', object: 'model', created: 1776766009709, owned_by: 'google', gemini_model: 'gemma-3-27b-it' },
  ]
  return c.json({ object: 'list', data: models })
})

app.post('/v1/chat/completions', async (c) => {
  if (!API_KEY) {
    return c.json({ error: 'GEMINI_API_KEY not configured' }, 500)
  }

  const body = await c.req.json()
  const { messages, model, stream, temperature, max_tokens } = body

  const requestedModel = model?.replace('gemini/', '').replace('models/', '') || 'gemini-2.5-flash'
  const isGemma = requestedModel.startsWith('gemma')

  const endpoint = isGemma ? `models/${requestedModel}` : `models/${requestedModel}`
  const useV2Endpoint = !isGemma

  const contents = messages.map((m: any) => ({
    role: m.role === 'assistant' ? 'model' : 'user',
    parts: [{ text: m.content }],
  }))

  const geminiPayload: any = {
    contents,
    generationConfig: {
      temperature: temperature ?? 0.9,
      maxOutputTokens: max_tokens ?? 2048,
    },
  }

  if (stream) {
    geminiPayload.generationConfig.enableStreaming = true
  }

  try {
    const url = `${BASE_URL}/${endpoint}?key=${API_KEY}`
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(geminiPayload),
    })

    if (!response.ok) {
      const err = await response.text()
      console.error(`[GEMMA-PROXY] ERROR ${response.status}: ${err}`)
      return c.json({ error: `Gemini API error: ${response.status}` }, response.status)
    }

    const data = await response.json()

    if (stream) {
      c.header('Content-Type', 'text/event-stream')
      c.header('Cache-Control', 'no-cache')
      c.header('Connection', 'keep-alive')

      let reply = ''
      if (data.candidates?.[0]?.content?.parts?.[0]?.text) {
        reply = data.candidates[0].content.parts[0].text
      }

      const chunk = JSON.stringify({
        id: 'chatcmpl-' + Date.now(),
        object: 'chat.completion.chunk',
        created: Math.floor(Date.now() / 1000),
        model: requestedModel,
        choices: [{
          index: 0,
          delta: { content: reply },
          finish_reason: 'stop',
        }],
      })

      const encoder = new TextEncoder()
      const chunks = [
        encoder.encode(`data: ${chunk}\n\n`),
      ]

        chunks.push(encoder.encode('data: [DONE]\n\n'))
      return c.body(chunks.reduce((a, b) => a + b))
    }

    let reply = ''
    if (data.candidates?.[0]?.content?.parts?.[0]?.text) {
      reply = data.candidates[0].content.parts[0].text
    }

    return c.json({
      id: 'chatcmpl-' + Date.now(),
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: requestedModel,
      choices: [{
        index: 0,
        message: { role: 'assistant', content: reply },
        finish_reason: 'stop',
      }],
      usage: {
        prompt_tokens: data.usageMetadata?.promptTokenCount || 0,
        completion_tokens: data.usageMetadata?.candidatesTokenCount || 0,
        total_tokens: (data.usageMetadata?.promptTokenCount || 0) + (data.usageMetadata?.candidatesTokenCount || 0),
      },
    })
  } catch (error) {
    console.error(`[GEMMA-PROXY] ERROR: ${error}`)
    return c.json({ error: String(error) }, 502)
  }
})

export default app