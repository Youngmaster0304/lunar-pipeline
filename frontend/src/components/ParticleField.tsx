import { useEffect, useRef } from 'react'

interface Star { x: number; y: number; r: number; a: number; da: number; s: number }

export default function ParticleField() {
  const canvas = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const c = canvas.current
    if (!c) return
    const ctx = c.getContext('2d')
    if (!ctx) return

    let frame: number
    const stars: Star[] = []
    let W = 0, H = 0

    function resize() {
      W = c!.width = window.innerWidth
      H = c!.height = window.innerHeight
    }
    resize()
    window.addEventListener('resize', resize)

    for (let i = 0; i < 120; i++) {
      stars.push({
        x: Math.random() * W,
        y: Math.random() * H,
        r: Math.random() * 1.5 + 0.3,
        a: Math.random() * Math.PI * 2,
        da: (Math.random() - 0.5) * 0.008,
        s: Math.random() * 0.3 + 0.1,
      })
    }

    function draw() {
      ctx!.clearRect(0, 0, W, H)
      for (const star of stars) {
        star.a += star.da
        const alpha = (Math.sin(star.a) + 1) * 0.35 + 0.1
        ctx!.beginPath()
        ctx!.arc(star.x, star.y, star.r, 0, Math.PI * 2)
        ctx!.fillStyle = `rgba(148, 163, 184, ${alpha})`
        ctx!.fill()
      }
      frame = requestAnimationFrame(draw)
    }
    frame = requestAnimationFrame(draw)

    return () => {
      cancelAnimationFrame(frame)
      window.removeEventListener('resize', resize)
    }
  }, [])

  return (
    <canvas
      ref={canvas}
      className="fixed inset-0 pointer-events-none"
      style={{ zIndex: 0 }}
    />
  )
}
