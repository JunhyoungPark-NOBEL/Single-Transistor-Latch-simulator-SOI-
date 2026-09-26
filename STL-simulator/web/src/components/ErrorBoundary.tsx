// Keeps a rendering error inside one panel/tab instead of blanking the whole app.
import { Component, type ErrorInfo, type ReactNode } from "react";

export class ErrorBoundary extends Component<{ children: ReactNode; label?: string; resetKey?: unknown }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("render error", this.props.label ?? "", error, info.componentStack);
  }
  componentDidUpdate(prev: { resetKey?: unknown }) {
    if (prev.resetKey !== this.props.resetKey && this.state.error) this.setState({ error: null });
  }
  render() {
    if (this.state.error)
      return (
        <div className="err-box" role="alert" style={{ margin: 8 }}>
          <strong>{this.props.label ?? "Render error"}:</strong> {this.state.error.message}
        </div>
      );
    return this.props.children;
  }
}
