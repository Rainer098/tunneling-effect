# Курсовая: туннельный эффект, 1D нестационарное уравнение Шрёдингера.
# i dψ/dt = -1/2 d²ψ/dx² + V(x) ψ, атомные единицы (ħ = m = 1).
# Считаем split-operator'ом, кинетику — через FFT.
#
# Запуск: python main.py          (параметры по умолчанию)
#         python main.py --ask    (спросит параметры)
#         python main.py --animate (плюс соберёт gif)

import argparse
from dataclasses import asdict

from src.solver import Params, SchrodingerSolver
from src.visualizer import Visualizer


def ask_params(default: Params) -> Params:
    """Спрашивает параметры, Enter — оставить по умолчанию."""
    values = asdict(default)
    keys = ["V0", "a", "E", "sigma", "x0", "L", "N", "dt", "steps"]
    print("Enter — оставить по умолчанию:")
    for key in keys:
        while True:
            try:
                raw = input(f"  {key} [{values[key]}]: ").strip()
            except EOFError:
                raw = ""
            if raw == "":
                break
            try:
                values[key] = int(raw) if key in ("N", "steps") else float(raw)
                break
            except ValueError:
                print(f"  '{raw}' не число, попробуй ещё раз")
    return Params(**values)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask", action="store_true",
                    help="спросить параметры с клавиатуры")
    ap.add_argument("--animate", action="store_true",
                    help="собрать gif с эволюцией")
    args = ap.parse_args()

    p = ask_params(Params()) if args.ask else Params()

    print("считаю...")
    s = SchrodingerSolver(p, save_every=10)
    s.run(verbose=True)

    T, R, P = s.balance()
    print(f"T = {T:.4f}   R = {R:.4f}   P = {P:.4f}")
    print(f"T + R     = {T + R:.4f}")
    print(f"T + R + P = {T + R + P:.4f}")

    # полный баланс должен сходиться строго
    assert abs(T + R + P - 1.0) < 0.01, f"T+R+P = {T+R+P}"
    # без P допуск шире: часть пакета ещё сидит в барьере
    assert abs(T + R - 1.0) < 0.02, f"T+R = {T+R}"

    vis = Visualizer(s.x, s.barrier)
    vis.plot_slices(s.times, s.densities, out_dir="results")

    if args.animate:
        vis.animate(s.times, s.densities, out_path="results/animation.gif")

    s.save("results")
    print("готово.")


if __name__ == "__main__":
    main()