export class GameSession {
  constructor() {
    this.gameId = null;
    this.gameMetadata = null;
    this.state = null;
    this.displayState = null;
    this.strategyData = null;
    this.catalogReady = false;
  }
}

export const gameSession = new GameSession();
