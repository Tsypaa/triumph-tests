import { afterEach, describe, expect, it, vi } from 'vitest'
import { removeBackground } from './api'

afterEach(() => vi.unstubAllGlobals())

describe('removeBackground', () => {
  it('returns a PNG blob', async () => {
    const output = new Blob(['png'], { type: 'image/png' })
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(output, {
      status: 200,
      headers: { 'content-type': 'image/png' },
    })))

    const result = await removeBackground(
      new File(['input'], 'photo.jpg', { type: 'image/jpeg' }),
      new AbortController().signal,
    )

    expect(result.type).toBe('image/png')
    expect(fetch).toHaveBeenCalledOnce()
    const request = vi.mocked(fetch).mock.calls[0]
    expect(request?.[0]).toBe('/api/remove-background')
    expect(request?.[1]?.method).toBe('POST')
  })

  it('preserves a structured API error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({
      error: { code: 'IMAGE_TOO_LARGE', message: 'Too many pixels.' },
    }), { status: 413, headers: { 'content-type': 'application/json' } })))

    const promise = removeBackground(
      new File(['input'], 'photo.png', { type: 'image/png' }),
      new AbortController().signal,
    )

    await expect(promise).rejects.toMatchObject({
      code: 'IMAGE_TOO_LARGE',
      status: 413,
      message: 'Too many pixels.',
    })
  })

  it('maps a network failure to a safe error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('private error')))
    const promise = removeBackground(
      new File(['input'], 'photo.webp', { type: 'image/webp' }),
      new AbortController().signal,
    )
    await expect(promise).rejects.toMatchObject({ code: 'NETWORK_ERROR' })
  })
})
