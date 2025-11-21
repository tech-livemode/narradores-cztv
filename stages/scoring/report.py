"""Console report helper for final scoring output."""

from model.scoreresult import FinalScore

def print_score_report(final_score: FinalScore):
    # Imprime um relatório formatado do score.
    print("\n" + "=" * 80)
    print("📊 RELATÓRIO DE AVALIAÇÃO DO NARRADOR ESPORTIVO")
    print("=" * 80)
    print(f"\n🎤 Stream ID: {final_score.stream_id}")
    print(f"⏱️  Duração Total: {final_score.total_duration:.2f}s")
    print(f"📝 Total de Segmentos: {final_score.total_segments}")

    print(f"\n{'=' * 80}")
    print("🎵 SCORE DE ÁUDIO (40%)")
    print(f"{'=' * 80}")
    audio = final_score.audio_score
    print(f"  • Dinâmica Vocal:      {audio.vocal_dynamics:.1f}/10")
    print(f"  • Ritmo da Fala:       {audio.speech_pacing:.1f}/10")
    print(f"  • Emoção (Áudio):      {audio.emotion_audio:.1f}/10")
    print(f"  ➜ TOTAL ÁUDIO:         {audio.total_score:.2f}/100")

    print(f"\n{'=' * 80}")
    print("📝 SCORE DE TEXTO (60%)")
    print(f"{'=' * 80}")
    text = final_score.text_score
    print(f"  • Emoção (Conteúdo):   {text.emotion_content:.1f}/10")
    print(f"  • Storytelling:        {text.storytelling:.1f}/10")
    print(f"  • Ritmo do Jogo:       {text.game_rhythm:.1f}/10")
    print(f"  • Resenha Apropriada:  {text.appropriate_fun:.1f}/10")
    print(f"  ➜ TOTAL TEXTO:         {text.total_score:.2f}/100")

    print(f"\n{'=' * 80}")
    print("🏆 SCORE FINAL")
    print(f"{'=' * 80}")
    print(f"  Score: {final_score.final_score:.2f}/100")
    print(f"  Classificação: {final_score.classification}")

    print(f"\n{'=' * 80}")
    print("✅ PONTOS FORTES")
    print(f"{'=' * 80}")
    for i, strength in enumerate(final_score.strengths, 1):
        print(f"  {i}. {strength}")

    print(f"\n{'=' * 80}")
    print("🎯 ÁREAS DE MELHORIA")
    print(f"{'=' * 80}")
    for i, improvement in enumerate(final_score.improvements, 1):
        print(f"  {i}. {improvement}")

    print(f"\n{'=' * 80}\n")


__all__ = ["print_score_report"]
