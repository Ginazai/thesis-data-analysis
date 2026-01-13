"""
Analizador de resultados de pruebas de visión artificial
Genera gráficos, métricas y tablas LaTeX para el reporte
INCLUYE: Extractor de logs del modo local (ESP32)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
import os
import json
from datetime import datetime, timedelta
from sklearn.metrics import confusion_matrix
import seaborn as sns

output_dir = "output_plots"
os.makedirs(output_dir, exist_ok=True)


class LocalLogParser:
    """Parser para logs del modo local ESP32"""
    
    def __init__(self, log_path: str, ground_truth_csv: str = "esp32_test.csv"):
        self.log_path = log_path
        self.ground_truth_csv = ground_truth_csv
        self.class_mapping = ['red', 'green', 'none', 'countdown_blank', 'countdown_green']
        self.ground_truth_df = None
        self.results = []
        
    def load_ground_truth(self):
        """Carga el CSV con las clases reales"""
        try:
            self.ground_truth_df = pd.read_csv(self.ground_truth_csv)
            print(f"✅ Ground truth cargado: {len(self.ground_truth_df)} imágenes")
            return True
        except Exception as e:
            print(f"❌ Error cargando ground truth: {e}")
            return False
    
    def parse_timestamp(self, ts_str: str) -> float:
        """Convierte T+HH:MM:SS.mmm a segundos totales"""
        try:
            ts_str = ts_str.replace("T+", "")
            parts = ts_str.split(":")
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            
            total_seconds = hours * 3600 + minutes * 60 + seconds
            return total_seconds
        except Exception as e:
            print(f"⚠️ Error parseando timestamp '{ts_str}': {e}")
            return None
    
    def get_class_index(self, class_name: str) -> int:
        """Obtiene el índice de una clase"""
        class_lower = class_name.lower().replace(" ", "_")
        if class_lower in self.class_mapping:
            return self.class_mapping.index(class_lower)
        return -1
    
    def get_class_from_index(self, index: int) -> str:
        """Obtiene nombre de clase desde índice"""
        if 0 <= index < len(self.class_mapping):
            return self.class_mapping[index]
        return "unknown"
    
    def parse_log(self):
        """Parsea el archivo de log completo"""
        if not self.load_ground_truth():
            return False
        
        print(f"\n📖 Parseando log: {self.log_path}")
        
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"❌ Error leyendo log: {e}")
            return False
        
        current_test = {}
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Extraer timestamp
            ts_match = re.match(r'(T\+\d{2}:\d{2}:\d{2}\.\d{3})', line)
            if not ts_match:
                continue
            
            timestamp_str = ts_match.group(1)
            timestamp = self.parse_timestamp(timestamp_str)
            
            # Testing image
            if "Testing image:" in line:
                match = re.search(r'Testing image: (.+?\.(?:jpg|JPG|jpeg|png))\s+\((\d+)/(\d+)\)', line)
                if match:
                    filepath = match.group(1)
                    filename = os.path.basename(filepath)
                    test_num = int(match.group(2))
                    
                    current_test = {
                        'imagen': filename,
                        'test_num': test_num,
                        'timestamp_start': timestamp,
                        'timestamp_start_str': timestamp_str
                    }
            
            # All data sent
            elif "All data sent" in line and current_test:
                match = re.search(r'All data sent \((\d+) bytes\)', line)
                if match:
                    current_test['bytes_sent'] = int(match.group(1))
                    current_test['timestamp_sent'] = timestamp
                    current_test['timestamp_sent_str'] = timestamp_str
            
            # JSON response
            elif line.startswith('T+') and '{' in line and '"status"' in line:
                try:
                    json_str = line.split(' | INFO | ')[1]
                    response = json.loads(json_str)
                    
                    if current_test and 'timestamp_sent' in current_test:
                        current_test['timestamp_response'] = timestamp
                        current_test['timestamp_response_str'] = timestamp_str
                        current_test['prediction'] = response.get('prediction', 'unknown')
                        current_test['confidence'] = response.get('confidence', 0.0)
                        current_test['inference_time_ms'] = response.get('inference_time_ms', 0)
                        current_test['detected'] = response.get('detected', False)
                        
                        # Calcular latencia total (desde start hasta response)
                        current_test['latencia_total_ms'] = (current_test['timestamp_response'] - current_test['timestamp_start']) * 1000
                        
                        # Calcular latencia de red (desde sent hasta response)
                        current_test['latencia_red_ms'] = (current_test['timestamp_response'] - current_test['timestamp_sent']) * 1000
                        
                        # Latencia de envío (desde start hasta sent)
                        current_test['latencia_envio_ms'] = (current_test['timestamp_sent'] - current_test['timestamp_start']) * 1000
                        
                        # Obtener clase real del ground truth
                        gt_row = self.ground_truth_df[
                            self.ground_truth_df['file'] == current_test['imagen']
                        ]
                        
                        if not gt_row.empty:
                            mode_index = int(gt_row.iloc[0]['mode'])
                            current_test['clase_real'] = self.get_class_from_index(mode_index)
                            current_test['clase_real_index'] = mode_index
                        else:
                            current_test['clase_real'] = 'unknown'
                            current_test['clase_real_index'] = -1
                            print(f"⚠️ No se encontró ground truth para: {current_test['imagen']}")
                        
                        # Mapear predicción
                        current_test['clase_pred'] = current_test['prediction'].lower().replace(" ", "_")
                        current_test['clase_pred_index'] = self.get_class_index(current_test['prediction'])
                        
                        # Calcular acierto
                        current_test['acierto'] = 1 if current_test['clase_real'] == current_test['clase_pred'] else 0
                        
                        # Guardar resultado
                        self.results.append(current_test.copy())
                        
                        # Reset para siguiente prueba
                        current_test = {}
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️ Error parseando JSON en línea {i+1}: {e}")
                except Exception as e:
                    print(f"⚠️ Error procesando respuesta en línea {i+1}: {e}")
        
        print(f"✅ Parseado completo: {len(self.results)} pruebas procesadas")
        return True
    
    def generate_results_csv(self, output_path: str = None):
        """Genera CSV con formato: imagen, clase_real, clase_pred, confianza, latencia_ms, acierto"""
        if not self.results:
            print("❌ No hay resultados para exportar")
            return None
        
        if output_path is None:
            output_path = os.path.join(output_dir, "resultados_modo_local.csv")
        
        df = pd.DataFrame(self.results)
        
        # Seleccionar y ordenar columnas - usar latencia_total_ms
        output_df = df[[
            'imagen', 
            'clase_real', 
            'clase_pred', 
            'confidence', 
            'latencia_total_ms',  # Cambiado aquí
            'acierto'
        ]].copy()
        
        # Renombrar para formato final
        output_df.columns = ['imagen', 'clase_real', 'clase_pred', 'confianza', 'latencia_ms', 'acierto']
        
        # Guardar
        output_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✅ CSV de resultados guardado: {output_path}")
        
        # Mostrar resumen
        print("\n📊 RESUMEN DE RESULTADOS:")
        print(f"   Total pruebas: {len(output_df)}")
        print(f"   Aciertos: {output_df['acierto'].sum()}")
        print(f"   Errores: {len(output_df) - output_df['acierto'].sum()}")
        print(f"   Precisión: {output_df['acierto'].mean() * 100:.2f}%")
        print(f"   Latencia media: {output_df['latencia_ms'].mean():.2f} ms")
        print(f"   Confianza media: {output_df['confianza'].mean():.4f}")
        
        return output_df
    
    def generate_logs_csv(self, output_path: str = None):
        """Genera CSV detallado con todos los datos parseados"""
        if not self.results:
            print("❌ No hay resultados para exportar")
            return None
        
        if output_path is None:
            output_path = os.path.join(output_dir, "logs_modo_local_detallado.csv")
        
        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✅ CSV de logs detallado guardado: {output_path}")
        
        return df
    
    def generate_confusion_matrix(self, output_path: str = None):
        """Genera y guarda la matriz de confusión"""
        if not self.results:
            print("❌ No hay resultados para generar matriz de confusión")
            return None
        
        if output_path is None:
            output_path = os.path.join(output_dir, "confusion_matrix_local.png")
        
        df = pd.DataFrame(self.results)
        df_valid = df[df['acierto'].notna()].copy()
        
        if df_valid.empty:
            print("❌ No hay datos válidos para matriz de confusión")
            return None
        
        
        # Obtener clases únicas en orden
        class_order = ['red', 'none', 'green', 'countdown_green', 'countdown_blank']
        
        # Generar matriz de confusión
        cm = confusion_matrix(
            df_valid['clase_real'], 
            df_valid['clase_pred'],
            labels=class_order
        )
        
        # Crear figura
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_order,
            yticklabels=class_order,
            cbar_kws={'label': 'Frecuencia'}
        )
        
        plt.title('Matriz de Confusión - Modo Local ESP32', fontsize=14, fontweight='bold')
        plt.ylabel('Clase Real (Ground Truth)', fontsize=12)
        plt.xlabel('Clase Predicha', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Guardar
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Matriz de confusión guardada: {output_path}")
        plt.close()
        
        return cm
    
    def calculate_local_metrics(self):
        """Calcula métricas específicas del modo local ESP32"""
        if not self.results:
            print("❌ No hay resultados para calcular métricas")
            return None
        
        df = pd.DataFrame(self.results)
        
        # Filtrar solo resultados válidos (con respuesta)
        df_valid = df[df['acierto'].notna()].copy()
        
        metrics = {
            'n_pruebas': len(df_valid),
            'aciertos': int(df_valid['acierto'].sum()),
            'errores': int(len(df_valid) - df_valid['acierto'].sum()),
            'precision_pct': round(df_valid['acierto'].mean() * 100, 2),
            
            # Latencia total (start -> response)
            'latencia_total_media_ms': round(df_valid['latencia_total_ms'].mean(), 2),
            'latencia_total_sd_ms': round(df_valid['latencia_total_ms'].std(), 2),
            'latencia_total_min_ms': round(df_valid['latencia_total_ms'].min(), 2),
            'latencia_total_max_ms': round(df_valid['latencia_total_ms'].max(), 2),
            'latencia_total_p95_ms': round(df_valid['latencia_total_ms'].quantile(0.95), 2),
            
            # Componentes de latencia
            'latencia_envio_media_ms': round(df_valid['latencia_envio_ms'].mean(), 2),
            'latencia_red_media_ms': round(df_valid['latencia_red_ms'].mean(), 2),
            'inference_time_media_ms': round(df_valid['inference_time_ms'].mean(), 2),
            
            # Confianza
            'confianza_media': round(df_valid['confidence'].mean(), 4),
            'confianza_sd': round(df_valid['confidence'].std(), 4),
            
            # Fallos de comunicación
            'total_intentos': len(df),
            'fallos_comunicacion': len(df) - len(df_valid),
            'tasa_fallos_pct': round((len(df) - len(df_valid)) / len(df) * 100, 2) if len(df) > 0 else 0
        }
        
        return metrics
    
    def print_metrics_summary(self, metrics):
        """Imprime resumen de métricas"""
        if not metrics:
            return
        
        print("\n" + "="*60)
        print("📊 MÉTRICAS DEL MODO LOCAL ESP32")
        print("="*60)
        
        print(f"\n✅ ACCURACY:")
        print(f"   Predicciones correctas: {metrics['aciertos']}/{metrics['n_pruebas']}")
        print(f"   Accuracy: {metrics['precision_pct']:.2f}%")
        
        print(f"\n⏱️  LATENCIA TOTAL:")
        print(f"   Media: {metrics['latencia_total_media_ms']:.2f} ms")
        print(f"   Desv. Est: {metrics['latencia_total_sd_ms']:.2f} ms")
        print(f"   Mínima: {metrics['latencia_total_min_ms']:.2f} ms")
        print(f"   Máxima: {metrics['latencia_total_max_ms']:.2f} ms")
        print(f"   P95: {metrics['latencia_total_p95_ms']:.2f} ms")
        
        print(f"\n🔧 DESGLOSE DE LATENCIA:")
        print(f"   Envío (ESP32): {metrics['latencia_envio_media_ms']:.2f} ms")
        print(f"   Inferencia (Servidor): {metrics['inference_time_media_ms']:.2f} ms")
        print(f"   Red (ida + vuelta): {metrics['latencia_red_media_ms']:.2f} ms")
        
        print(f"\n📡 COMUNICACIÓN:")
        print(f"   Intentos totales: {metrics['total_intentos']}")
        print(f"   Respuestas exitosas: {metrics['n_pruebas']}")
        print(f"   Fallos: {metrics['fallos_comunicacion']}")
        print(f"   Tasa de fallos: {metrics['tasa_fallos_pct']:.2f}%")
        
        print(f"\n🎯 CONFIANZA:")
        print(f"   Media: {metrics['confianza_media']:.4f}")
        print(f"   Desv. Est: {metrics['confianza_sd']:.4f}")

class TestAnalyzer:
    def __init__(self, csv_path: str = "resultados_pruebas_20260104_161812.csv"):
        self.csv_path = csv_path
        self.df = None
        self.metrics = {}
        
    def load_data(self):
        """Carga el CSV de resultados"""
        try:
            try:
                self.df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
            except:
                self.df = pd.read_csv(self.csv_path, encoding='utf-8')
            
            print(f"✅ CSV cargado: {len(self.df)} registros")
            print(f"📋 Columnas: {list(self.df.columns)}")
            
            # Convertir latencia y t_total_ms
            if 'latencia' in self.df.columns:
                self.df['latencia_ms'] = self.df['latencia'].apply(self.mmss_to_seconds)
                
            if 't_total_ms' in self.df.columns:
                self.df['t_total_ms_numeric'] = self.df['t_total_ms'].apply(self.mmss_to_seconds)
            
            # Convertir confianza a numérico
            if 'confianza' in self.df.columns:
                self.df['confianza'] = pd.to_numeric(self.df['confianza'], errors='coerce')
            
            # Procesar DEPTH
            self.df['distancia_verdad_cm'] = self.df.apply(
                lambda row: self.extract_depth_ground_truth(row) 
                if row.get('tipo_evento') == 'DEPTH' else None, 
                axis=1
            )
            
            self.df['distancia_estimada_cm'] = self.df.apply(
                lambda row: self.extract_distance(row.get('objeto_predicho')) 
                if row.get('tipo_evento') == 'DEPTH' else None, 
                axis=1
            )
            
            # Calcular diferencias de distancia
            self.df['diferencia_distancia_pct'] = self.df.apply(
                lambda row: self.calculate_distance_error(
                    row.get('distancia_verdad_cm'), 
                    row.get('distancia_estimada_cm')
                ), axis=1
            )
            
            self.df['clasificacion_distancia'] = self.df['diferencia_distancia_pct'].apply(
                self.classify_distance_error
            )
            
            # Procesar OCR
            self.df['texto_verdad'] = self.df.apply(
                lambda row: self.extract_ocr_ground_truth(row) 
                if row.get('tipo_evento') == 'OCR' else None, 
                axis=1
            )
            
            self.df['texto_predicho'] = self.df.apply(
                lambda row: self.extract_ocr_predicted(row) 
                if row.get('tipo_evento') == 'OCR' else None, 
                axis=1
            )
            
            return True
        except Exception as e:
            print(f"❌ Error al cargar datos: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def extract_distance(self, dist_str) -> float:
        """Extrae valor numérico de distancia en cm"""
        if pd.isna(dist_str):
            return None
        
        try:
            dist_str = str(dist_str).strip()
            if not dist_str or dist_str in ['', 'nan', 'None']:
                return None
            
            match = re.search(r'(\d+\.?\d*)\s*cm', dist_str, re.IGNORECASE)
            if match:
                return float(match.group(1))
            
            try:
                return float(dist_str)
            except:
                return None
        except Exception:
            return None
    
    def extract_depth_ground_truth(self, row) -> float:
        """Extrae la distancia verdad de DEPTH"""
        if pd.notna(row.get('objeto_verdad')):
            obj_verdad = str(row['objeto_verdad']).strip()
            if obj_verdad and obj_verdad.lower() != 'nan':
                dist = self.extract_distance(obj_verdad)
                if dist is not None:
                    return dist
        
        if pd.notna(row.get('distancia_m')):
            try:
                return float(row['distancia_m']) * 100
            except:
                pass
        
        escenario = row.get('escenario', '')
        if pd.notna(escenario):
            escenario_str = str(escenario)
            match = re.search(r'(\d+\.?\d*)\s*cm', escenario_str, re.IGNORECASE)
            if match:
                return float(match.group(1))
            match = re.search(r'(\d+\.?\d*)\s*m\b', escenario_str, re.IGNORECASE)
            if match:
                return float(match.group(1)) * 100
        
        return None
    
    def extract_ocr_ground_truth(self, row) -> str:
        """Extrae el texto verdad de OCR"""
        if pd.notna(row.get('objeto_verdad')):
            obj_verdad = str(row['objeto_verdad']).strip()
            if obj_verdad and obj_verdad.lower() not in ['nan', '', 'none']:
                return obj_verdad
        return None
    
    def extract_ocr_predicted(self, row) -> str:
        """Extrae el texto predicho de OCR"""
        notas = row.get('notas')
        if pd.notna(notas):
            notas_str = str(notas)
            match = re.search(r'result:\s*["\']([^"\']+)["\']', notas_str, re.IGNORECASE | re.DOTALL)
            if match:
                resultado = match.group(1).strip()
                resultado = ' '.join(resultado.split())
                return resultado
        
        if pd.notna(row.get('objeto_predicho')):
            obj_pred = str(row['objeto_predicho']).strip()
            if not re.search(r'\d+\.?\d*\s*cm', obj_pred, re.IGNORECASE):
                return obj_pred
        
        return None
    
    def calculate_distance_error(self, real, estimada) -> float:
        """Calcula error porcentual"""
        if pd.isna(real) or pd.isna(estimada) or real == 0:
            return None
        
        try:
            error_pct = (abs(real - estimada) / real) * 100
            return round(error_pct, 2)
        except:
            return None
    
    def classify_distance_error(self, error_pct) -> str:
        """Clasifica error de distancia"""
        if pd.isna(error_pct):
            return None
        
        if error_pct < 20:
            return 'Acierto'
        elif error_pct <= 50:
            return 'Error parcial'
        else:
            return 'Fallo'
    
    def mmss_to_seconds(self, time_str: str) -> float:
        """Convierte MM:SS.s o HH:MM:SS.s a segundos"""
        if pd.isna(time_str):
            return None

        try:
            time_str = str(time_str).strip()
            if time_str == "":
                return None

            is_negative = time_str.startswith("-")
            if is_negative:
                time_str = time_str[1:]

            parts = time_str.split(":")

            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = float(parts[1])
                total_seconds = minutes * 60 + seconds
            elif len(parts) == 3:
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
                total_seconds = hours * 3600 + minutes * 60 + seconds
            else:
                return None

            return -total_seconds if is_negative else total_seconds

        except Exception:
            return None
    
    def calculate_text_similarity(self, ground_truth: str, predicted: str) -> float:
        """Calcula similitud usando distancia de Levenshtein"""
        gt = str(ground_truth).lower().strip()
        pred = str(predicted).lower().strip()
        
        if gt == pred:
            return 1.0
        
        if not gt or not pred:
            return 0.0
        
        distance = self.levenshtein_distance(gt, pred)
        max_len = max(len(gt), len(pred))
        similarity = 1.0 - (distance / max_len)
        
        return max(0.0, similarity)
    
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Distancia de Levenshtein"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def generate_report(self):
        """Genera reporte completo (híbrido)"""
        if not self.load_data():
            return
        
        print("\n" + "="*60)
        print("📋 REPORTE DE ANÁLISIS - MODO HÍBRIDO")
        print("="*60)
        print("(Implementación pendiente de métricas híbridas)")


if __name__ == "__main__":
    print("="*60)
    print("🚀 ANALIZADOR DE PRUEBAS DE VISIÓN ARTIFICIAL")
    print("="*60)
    
    # === PROCESAR LOGS DEL MODO LOCAL ===
    print("\n📖 PROCESANDO LOGS DEL MODO LOCAL ESP32...")
    
    local_parser = LocalLogParser(
        log_path="./logs/ESP - 2 - 4.1 - 4.2.log",
        ground_truth_csv="esp32_test.csv"
    )

    if local_parser.parse_log():
        # Generar CSVs
        local_parser.generate_results_csv()
        local_parser.generate_logs_csv()
        
        # Calcular y mostrar métricas
        metrics = local_parser.calculate_local_metrics()
        local_parser.print_metrics_summary(metrics)
        
        # Generar matriz de confusión
        local_parser.generate_confusion_matrix()
        
        print("\n✅ Análisis del modo local completado")
    
    # Analizar resultados híbridos (si existe el CSV)
    if os.path.exists("resultados_pruebas_20260104_161812.csv"):
        print("\n📊 ANALIZANDO RESULTADOS DEL MODO HÍBRIDO...")
        analyzer = TestAnalyzer(csv_path="resultados_pruebas_20260104_161812.csv")
        analyzer.generate_report()