import { ChangeEvent, DragEvent, useEffect, useRef, useState } from 'react'
import { RemoveBackgroundError, removeBackground } from './api'

const ACCEPTED_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp'])
const MAX_CLIENT_FILE_BYTES = 15 * 1024 * 1024

type Phase = 'empty' | 'ready' | 'processing' | 'done' | 'error'

function formatSize(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} КБ`
  return `${(bytes / (1024 * 1024)).toFixed(1)} МБ`
}

function UploadIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" width="32" height="32">
      <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 14v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4" />
    </svg>
  )
}

function ImagePanel({ label, source, transparent = false }: { label: string; source: string; transparent?: boolean }) {
  return (
    <figure className={`image-panel${transparent ? ' checkerboard' : ''}`}>
      <figcaption>{label}</figcaption>
      <div className="image-frame">
        <img src={source} alt={label} />
      </div>
    </figure>
  )
}

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [resultUrl, setResultUrl] = useState<string | null>(null)
  const [resultBlob, setResultBlob] = useState<Blob | null>(null)
  const [phase, setPhase] = useState<Phase>('empty')
  const [message, setMessage] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const abortRef = useRef<AbortController | null>(null)
  const previewUrlRef = useRef<string | null>(null)
  const resultUrlRef = useRef<string | null>(null)

  useEffect(() => () => {
    abortRef.current?.abort()
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current)
    if (resultUrlRef.current) URL.revokeObjectURL(resultUrlRef.current)
  }, [])

  const releaseUrls = () => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current)
    if (resultUrlRef.current) URL.revokeObjectURL(resultUrlRef.current)
    previewUrlRef.current = null
    resultUrlRef.current = null
  }

  const selectFile = (nextFile: File | undefined) => {
    if (!nextFile) return
    abortRef.current?.abort()
    releaseUrls()
    setResultUrl(null)
    setResultBlob(null)

    if (!ACCEPTED_TYPES.has(nextFile.type)) {
      setFile(null)
      setPreviewUrl(null)
      setPhase('error')
      setMessage('Выберите изображение JPEG, PNG или WebP.')
      return
    }
    if (nextFile.size > MAX_CLIENT_FILE_BYTES) {
      setFile(null)
      setPreviewUrl(null)
      setPhase('error')
      setMessage('Файл больше 15 МБ. Выберите изображение меньшего размера.')
      return
    }
    setFile(nextFile)
    const nextPreviewUrl = URL.createObjectURL(nextFile)
    previewUrlRef.current = nextPreviewUrl
    setPreviewUrl(nextPreviewUrl)
    setPhase('ready')
    setMessage(null)
  }

  const onInput = (event: ChangeEvent<HTMLInputElement>) => {
    selectFile(event.target.files?.[0])
    event.target.value = ''
  }

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragging(false)
    selectFile(event.dataTransfer.files[0])
  }

  const processImage = async () => {
    if (!file) return
    const controller = new AbortController()
    abortRef.current = controller
    setPhase('processing')
    setMessage(null)
    try {
      const blob = await removeBackground(file, controller.signal)
      if (controller.signal.aborted) return
      if (resultUrlRef.current) URL.revokeObjectURL(resultUrlRef.current)
      const nextResultUrl = URL.createObjectURL(blob)
      resultUrlRef.current = nextResultUrl
      setResultBlob(blob)
      setResultUrl(nextResultUrl)
      setPhase('done')
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        setPhase(file ? 'ready' : 'empty')
        setMessage(null)
        return
      }
      setPhase('error')
      setMessage(
        error instanceof RemoveBackgroundError
          ? error.message
          : 'Не удалось обработать изображение. Попробуйте ещё раз.',
      )
    } finally {
      if (abortRef.current === controller) abortRef.current = null
    }
  }

  const reset = () => {
    abortRef.current?.abort()
    releaseUrls()
    setFile(null)
    setPreviewUrl(null)
    setResultUrl(null)
    setResultBlob(null)
    setMessage(null)
    setPhase('empty')
  }

  const cancel = () => abortRef.current?.abort()

  const download = () => {
    if (!resultBlob || !resultUrl) return
    const link = document.createElement('a')
    const baseName = file?.name.replace(/\.[^.]+$/, '') || 'image'
    link.href = resultUrl
    link.download = `${baseName}-no-bg.png`
    document.body.appendChild(link)
    link.click()
    link.remove()
  }

  return (
    <main>
      <header className="site-header">
        <a className="brand" href="/" aria-label="Clearcut — главная">
          <span className="brand-mark" />
          Clearcut
        </a>
        <span className="privacy-note">Изображения обрабатываются на нашем сервере</span>
      </header>

      <section className="hero">
        <p className="eyebrow">Точное выделение объекта</p>
        <h1>Уберите фон.<br />Оставьте главное.</h1>
        <p className="subtitle">Загрузите фотографию — получите готовый PNG с прозрачным фоном.</p>
      </section>

      <section className="workspace" aria-live="polite">
        {phase === 'empty' || (phase === 'error' && !file) ? (
          <div
            className={`dropzone${dragging ? ' is-dragging' : ''}`}
            onDragEnter={(event) => { event.preventDefault(); setDragging(true) }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            <div className="upload-icon"><UploadIcon /></div>
            <h2>Перетащите изображение сюда</h2>
            <p>или выберите файл с устройства</p>
            <button className="primary" type="button" onClick={() => inputRef.current?.click()}>
              Выбрать изображение
            </button>
            <span className="file-hint">JPEG, PNG или WebP · до 15 МБ</span>
          </div>
        ) : null}

        {(phase === 'ready' || phase === 'processing' || (phase === 'error' && file)) && previewUrl ? (
          <div className="preview-card">
            <ImagePanel label="Исходное изображение" source={previewUrl} />
            <div className="preview-meta">
              <div>
                <strong>{file?.name}</strong>
                <span>{file ? formatSize(file.size) : null}</span>
              </div>
              {phase === 'processing' ? (
                <div className="processing-block">
                  <span className="spinner" aria-hidden="true" />
                  <div><strong>Удаляем фон…</strong><span>Сложные контуры требуют немного времени</span></div>
                  <button className="text-button" type="button" onClick={cancel}>Отменить</button>
                </div>
              ) : (
                <div className="action-row">
                  <button className="secondary" type="button" onClick={() => inputRef.current?.click()}>Заменить</button>
                  <button className="primary" type="button" onClick={processImage}>Удалить фон</button>
                </div>
              )}
            </div>
          </div>
        ) : null}

        {phase === 'done' && previewUrl && resultUrl ? (
          <div className="result-card">
            <div className="comparison">
              <ImagePanel label="До" source={previewUrl} />
              <ImagePanel label="После" source={resultUrl} transparent />
            </div>
            <div className="result-actions">
              <div><strong>Готово</strong><span>PNG с прозрачным фоном</span></div>
              <div className="action-row">
                <button className="secondary" type="button" onClick={reset}>Новое изображение</button>
                <button className="primary" type="button" onClick={download}>Скачать PNG</button>
              </div>
            </div>
          </div>
        ) : null}

        {message ? (
          <div className="error-banner" role="alert">
            <span aria-hidden="true">!</span>
            <p>{message}</p>
            {file ? <button type="button" onClick={processImage}>Повторить</button> : null}
          </div>
        ) : null}
      </section>

      <input ref={inputRef} className="visually-hidden" type="file" accept="image/jpeg,image/png,image/webp" onChange={onInput} />

      <footer>
        <span>BiRefNet · локальный inference</span>
        <span>JPEG · PNG · WebP</span>
      </footer>
    </main>
  )
}
