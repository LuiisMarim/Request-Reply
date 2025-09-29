from __future__ import annotations

import argparse
import logging
import sys

from application.use_cases.validate_data import ValidateDataUseCase
from application.use_cases.extract_features import ExtractFeaturesUseCase
from application.use_cases.train import TrainUseCase
from application.use_cases.infer import InferUseCase
from application.use_cases.benchmark import BenchmarkUseCase
from application.dto.validate_data_dto import ValidateDataRequest
from application.dto.extract_features_dto import ExtractFeaturesRequest
from application.dto.train_dto import TrainRequest
from application.dto.infer_dto import InferRequest
from application.dto.benchmark_dto import BenchmarkRequest
from infrastructure.data.validators import DataValidator
from infrastructure.vision.au_extractor_openface import OpenFaceAUExtractor

logger = logging.getLogger(__name__)


def main() -> None:
    """
    CLI para orquestração do pipeline TEA Biomarker.

    Subcomandos disponíveis:
      - validate-data
      - extract-features
      - train
      - infer
      - benchmark
    """
    parser = argparse.ArgumentParser(description="CLI do pipeline TEA Biomarker")
    subparsers = parser.add_subparsers(dest="command")

    # validate-data
    validate_p = subparsers.add_parser("validate-data", help="Validação de dados")
    validate_p.add_argument("--input", required=True)
    validate_p.add_argument("--out", required=True)
    validate_p.add_argument("--report", required=True)

    # extract-features
    extract_p = subparsers.add_parser("extract-features", help="Extração de features")
    extract_p.add_argument("--input", required=True)
    extract_p.add_argument("--out", required=True)
    extract_p.add_argument("--profile", default="medium")

    # train
    train_p = subparsers.add_parser("train", help="Treino do modelo")
    train_p.add_argument("--features", required=True)
    train_p.add_argument("--models-dir", required=True)
    train_p.add_argument("--profile", default="medium")
    train_p.add_argument("--protocol", choices=["loso", "kfold"], default="loso")
    train_p.add_argument("--enable-xai", action="store_true")
    train_p.add_argument("--audit-fairness", action="store_true")

    # infer
    infer_p = subparsers.add_parser("infer", help="Inferência")
    infer_p.add_argument("--input", required=True)
    infer_p.add_argument("--models-dir", required=True)
    infer_p.add_argument("--out", required=True)

    # benchmark
    bench_p = subparsers.add_parser("benchmark", help="Benchmark de desempenho")
    bench_p.add_argument("--profile", default="medium")
    bench_p.add_argument("--iterations", type=int, default=100)

    args = parser.parse_args()

    try:
        if args.command == "validate-data":
            if not args.input.strip() or not args.out.strip() or not args.report.strip():
                raise ValueError("Parâmetros inválidos para validate-data.")
            req = ValidateDataRequest(args.input, args.out, args.report)
            validator = DataValidator()
            res = ValidateDataUseCase(None, validator).execute(req)
            print(f"✅ Dados validados. Relatório: {res.report_path}")
            logger.info("Validação concluída com relatório em %s", res.report_path)
            
        elif args.command == "extract-features":
            if not args.input.strip() or not args.out.strip():
                raise ValueError("Parâmetros inválidos para extract-features.")
            req = ExtractFeaturesRequest(args.input, args.out, args.profile)
            extractor = OpenFaceAUExtractor(
                openface_bin=r"C:\Users\luisa\OneDrive\Documentos\Request-Reply\project\OpenFace\FeatureExtraction.exe"
            )
            res = ExtractFeaturesUseCase(extractor).execute(req)
            print(f"✅ Features extraídas em {res.features_dir}")
            logger.info("Extração de features concluída em %s", res.features_dir)

        elif args.command == "train":
            if not args.features.strip() or not args.models_dir.strip():
                raise ValueError("Parâmetros inválidos para train.")
            req = TrainRequest(
                features_path=args.features,
                models_dir=args.models_dir,
                protocol=args.protocol,
                profile=args.profile,
                enable_xai=args.enable_xai,
                audit_fairness=args.audit_fairness,
            )
            res = TrainUseCase().execute(req)
            print(f"✅ Treino concluído. Modelos em {res.models_dir}")
            logger.info("Treinamento concluído. Modelos em %s", res.models_dir)

        elif args.command == "infer":
            if not args.input.strip() or not args.models_dir.strip() or not args.out.strip():
                raise ValueError("Parâmetros inválidos para infer.")
            req = InferRequest(args.input, args.models_dir, args.out)
            res = InferUseCase().execute(req)
            print(f"✅ Inferência concluída. Relatórios em {res.output_dir}")
            logger.info("Inferência concluída. Relatórios em %s", res.output_dir)

        elif args.command == "benchmark":
            if args.iterations <= 0 or not args.profile.strip():
                raise ValueError("Parâmetros inválidos para benchmark.")
            req = BenchmarkRequest(args.profile, args.iterations)
            res = BenchmarkUseCase().execute(req)
            print(f"✅ Benchmark concluído. Tempo médio: {res.avg_time:.3f}s")
            logger.info("Benchmark concluído. Tempo médio: %.3fs", res.avg_time)

        else:
            parser.print_help()

    except Exception as e:
        logger.exception("Falha crítica na execução do comando: %s", str(e))
        print(f"❌ Falha: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
