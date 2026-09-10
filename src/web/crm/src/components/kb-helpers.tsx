/* Переэкспорт UI-примитивов для экрана «База знаний» */
export { Badge, Btn, Card, Field, inputCls, useToast } from './ui';

export function PostThumbWrap({ hue, size = 84 }: { hue: number; size?: number }) {
  return (
    <div className="relative rounded-xl overflow-hidden shrink-0 shadow-sm border border-gray-100"
      style={{ width: size, height: size, background: `linear-gradient(140deg, hsl(${hue} 72% 84%), hsl(${hue + 42} 62% 60%))` }}>
      <div className="absolute inset-0" style={{ background: 'radial-gradient(circle at 30% 25%, rgba(255,255,255,.55), transparent 55%)' }} />
      <div className="absolute bottom-0 inset-x-0 h-1/3" style={{ background: 'linear-gradient(transparent, rgba(17,24,39,.22))' }} />
    </div>
  );
}
