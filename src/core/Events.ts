/** Typed pub/sub used to keep gameplay, audio, UI and VFX decoupled. */

import type { Vector3 } from 'three';

export interface GameEvents {
  /** Ball left a shooter's hands. */
  shotReleased: { quality: number; distance: number; three: boolean; shooter: number };
  /** Ball passed cleanly through the net. */
  scored: { points: number; team: number; shooter: number; swish: boolean; assisted: number | null };
  /** Shot hit iron and came off. */
  missed: { team: number; shooter: number; rimContact: boolean };
  rimContact: { speed: number; position: Vector3 };
  boardContact: { speed: number; position: Vector3 };
  netSwish: { position: Vector3 };
  floorBounce: { speed: number; position: Vector3 };
  dribble: { speed: number; position: Vector3 };
  rebound: { team: number; player: number; offensive: boolean };
  steal: { team: number; player: number };
  block: { team: number; player: number };
  possessionChanged: { team: number };
  dunk: { player: number; power: number };
  layup: { player: number };
  crossover: { player: number; broke: boolean };
  sneakerSqueak: { position: Vector3; intensity: number };
  jumpLand: { position: Vector3; force: number };
  clockTick: { seconds: number; shotClock: number };
  quarterEnd: { quarter: number };
  gameEnd: { home: number; away: number };
  cameraShake: { amount: number; duration: number };
  hudToast: { text: string; kind: 'good' | 'bad' | 'neutral' | 'hype' };
  qualityChanged: { tier: string; renderScale: number };
}

type Handler<T> = (payload: T) => void;

export class EventBus {
  private map = new Map<string, Set<Handler<never>>>();

  on<K extends keyof GameEvents>(key: K, fn: Handler<GameEvents[K]>): () => void {
    let set = this.map.get(key as string);
    if (!set) this.map.set(key as string, (set = new Set()));
    set.add(fn as Handler<never>);
    return () => set!.delete(fn as Handler<never>);
  }

  once<K extends keyof GameEvents>(key: K, fn: Handler<GameEvents[K]>): () => void {
    const off = this.on(key, (p) => {
      off();
      fn(p);
    });
    return off;
  }

  emit<K extends keyof GameEvents>(key: K, payload: GameEvents[K]): void {
    const set = this.map.get(key as string);
    if (!set) return;
    for (const fn of set) {
      try {
        (fn as Handler<GameEvents[K]>)(payload);
      } catch (err) {
        console.error(`[events] handler for "${String(key)}" threw`, err);
      }
    }
  }

  clear(): void {
    this.map.clear();
  }
}

export const bus = new EventBus();
