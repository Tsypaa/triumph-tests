export type ApiErrorPayload = {
  error?: {
    code?: string
    message?: string
  }
}

export class RemoveBackgroundError extends Error {
  readonly code: string
  readonly status: number | null

  constructor(message: string, code = 'NETWORK_ERROR', status: number | null = null) {
    super(message)
    this.name = 'RemoveBackgroundError'
    this.code = code
    this.status = status
  }
}

async function readApiError(response: Response): Promise<RemoveBackgroundError> {
  let payload: ApiErrorPayload | null = null
  try {
    payload = (await response.json()) as ApiErrorPayload
  } catch {
    // Reverse proxies may return non-JSON errors. Keep the public fallback safe.
  }
  return new RemoveBackgroundError(
    payload?.error?.message ?? 'Сервис не смог обработать изображение.',
    payload?.error?.code ?? 'API_ERROR',
    response.status,
  )
}

export async function removeBackground(file: File, signal: AbortSignal): Promise<Blob> {
  const form = new FormData()
  form.append('file', file)

  let response: Response
  try {
    response = await fetch('/api/remove-background', {
      method: 'POST',
      body: form,
      signal,
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw error
    }
    throw new RemoveBackgroundError(
      'Не удалось связаться с сервисом. Проверьте соединение и попробуйте снова.',
    )
  }

  if (!response.ok) {
    throw await readApiError(response)
  }
  const contentType = response.headers.get('content-type')?.split(';', 1)[0]
  if (contentType !== 'image/png') {
    throw new RemoveBackgroundError('Сервис вернул неожиданный формат результата.', 'INVALID_RESPONSE')
  }
  return response.blob()
}
