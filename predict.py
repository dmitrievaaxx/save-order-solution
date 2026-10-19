"""Оценка риска заказа и главная причина — эталонное решение этапа 6 «Спасти заказ».

    python predict.py --input заказы.csv --output result.csv

Результат: order_id, risk_score (0…1), main_reason. На плохих данных — сообщение,
что именно не так, код выхода 1 и никакого файла результата.
"""

import argparse
import sys

import numpy as np
import pandas as pd

COLUMNS = (
    "order_id", "chain", "region", "order_hour", "basket_items", "promised_duration_min",
    "current_status", "minutes_in_status", "minutes_since_last_event", "courier_assigned",
    "assignments_count", "ready_for_pickup", "courier_wait_min", "support_contacts_count",
    "current_delay_min",
)
NUMBERS = (
    "order_hour", "basket_items", "promised_duration_min", "minutes_in_status",
    "minutes_since_last_event", "courier_assigned", "assignments_count", "ready_for_pickup",
    "courier_wait_min", "support_contacts_count", "current_delay_min",
)
MINUTES = ("minutes_in_status", "minutes_since_last_event", "courier_wait_min", "current_delay_min")
# Пропуски здесь — не ошибка: товары не посчитаны, событий не было, курьера ещё нет.
MAY_BE_EMPTY = ("basket_items", "minutes_since_last_event", "courier_wait_min")
STATUSES = {"created", "assembling", "searching_courier", "courier_assigned", "in_delivery"}


class BadOrders(ValueError):
    """Данные, на которых оценивать нельзя; текст объясняет, что не так."""


def check(orders: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in COLUMNS if c not in orders.columns]
    if missing:
        raise BadOrders("Не хватает колонок: " + ", ".join(missing))
    if orders.empty:
        raise BadOrders("В файле нет заказов")
    repeated = orders["order_id"][orders["order_id"].duplicated()]
    if len(repeated):
        raise BadOrders(f"order_id встречается дважды: {repeated.iloc[0]}")
    orders = orders.copy()
    for column in NUMBERS:
        values = pd.to_numeric(orders[column], errors="coerce")
        broken = values.isna() & orders[column].notna()
        if broken.any():
            row = int(np.flatnonzero(broken.to_numpy())[0]) + 2  # строка файла: заголовок — первая
            raise BadOrders(f"В поле {column}, строка {row}: «{orders[column][broken].iloc[0]}» — не число")
        empty = values.isna()
        if empty.any() and column not in MAY_BE_EMPTY:
            raise BadOrders(f"В поле {column} пусто у заказа {orders['order_id'][empty].iloc[0]}")
        orders[column] = values
    for column in MINUTES:
        negative = orders[column] < 0
        if negative.any():
            raise BadOrders(
                f"В поле {column} отрицательные минуты у заказа {orders['order_id'][negative].iloc[0]}"
            )
    unknown = ~orders["current_status"].isin(STATUSES)
    if unknown.any():
        raise BadOrders(f"Неизвестный статус: {orders['current_status'][unknown].iloc[0]}")
    return orders


def score(orders: pd.DataFrame) -> pd.DataFrame:
    """Чем больше задержка и чем дольше заказ стоит в статусе, тем выше риск."""
    waiting = (orders["courier_assigned"] == 0) & (orders["current_status"] == "searching_courier")
    raw = (
        orders["current_delay_min"]
        + 0.5 * orders["minutes_in_status"]
        + 10 * orders["support_contacts_count"]
        + 10 * waiting
    )
    reason = np.select(
        [
            orders["courier_assigned"] == 0,
            orders["assignments_count"] > 1,
            orders["current_delay_min"] > 0,
            orders["support_contacts_count"] > 0,
            orders["courier_wait_min"].fillna(0) > 10,
        ],
        ["courier_not_assigned", "courier_reassigned", "already_delayed", "support_contacted",
         "courier_waiting"],
        default="long_in_status",
    )
    return pd.DataFrame(
        {"order_id": orders["order_id"], "risk_score": (raw / (raw + 30)).round(6),
         "main_reason": reason}
    )


def predict_one(order: dict) -> dict:
    """Один заказ словарём — как строка таблицы; пропуски — NaN."""
    result = score(check(pd.DataFrame([order]))).iloc[0]
    return {"risk_score": float(result["risk_score"]), "main_reason": str(result["main_reason"])}


def main() -> int:
    parser = argparse.ArgumentParser(description="Оценка риска заказов")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        orders = check(pd.read_csv(args.input))
    except BadOrders as error:
        print(f"Ошибка в данных: {error}", file=sys.stderr)
        return 1
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        print(f"Не удалось прочитать {args.input}: {error}", file=sys.stderr)
        return 1
    score(orders).to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
