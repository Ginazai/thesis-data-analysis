"""
Analizador de resultados de pruebas de visión artificial
Genera gráficos y métricas de precisión, latencia y errores
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class TestAnalyzer:
    def __init__(self, csv_path: str = "resultados_pruebas.csv"):
        self.csv_path = csv_path
        self.df = None
        
    def load_data(self):
        """Carga el CSV de resultados"""
        try:
            # Intentar con UTF-8-sig primero (con BOM), luego UTF-8
            try:
                self.df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
            except:
                self.df = pd.read_csv(self.csv_path, encoding='utf-8')
            
            # Convertir latencia y t_total_ms de formato HH:MM:SS.mmm a milisegundos para análisis
            if 'latencia' in self.df.columns:
                self.df['latencia_ms'] = self.df['latencia'].apply(self.hhmmss_to_ms)
                latencia_validos = self.df['latencia_ms'].notna().sum()
                print(f"   📊 Latencia: {latencia_validos} valores convertidos")
                
            if 't_total_ms' in self.df.columns:
                self.df['t_total_ms_numeric'] = self.df['t_total_ms'].apply(self.hhmmss_to_ms)
                t_total_validos = self.df['t_total_ms_numeric'].notna().sum()
                print(f"   📊 t_total_ms: {t_total_validos} valores convertidos")
            
            print(f"✅ Datos cargados: {len(self.df)} registros")
            return True
        except FileNotFoundError:
            print(f"❌ Error: No se encontró el archivo {self.csv_path}")
            return False
        except Exception as e:
            print(f"❌ Error al cargar datos: {str(e)}")
            return False
    
    def hhmmss_to_ms(self, time_str: str) -> float:
        """Convierte formato HH:MM:SS.mmm a milisegundos"""
        if pd.isna(time_str) or time_str == '' or time_str is None:
            return None
        
        try:
            # Convertir a string si no lo es
            time_str = str(time_str).strip()
            
            # Formato: HH:MM:SS.mmm o puede tener negativos
            is_negative = time_str.startswith('-')
            if is_negative:
                time_str = time_str[1:]  # Remover el signo negativo temporalmente
            
            parts = time_str.split(':')
            if len(parts) != 3:
                return None
                
            hours = int(parts[0])
            minutes = int(parts[1])
            
            # Manejar segundos y milisegundos
            sec_parts = parts[2].split('.')
            seconds = int(sec_parts[0])
            milliseconds = int(sec_parts[1]) if len(sec_parts) > 1 else 0
            
            total_ms = (hours * 3600 + minutes * 60 + seconds) * 1000 + milliseconds
            
            return float(total_ms) * (-1 if is_negative else 1)
        except Exception as e:
            print(f"⚠️ Error convirtiendo '{time_str}': {e}")
            return None
    
    def calculate_precision(self):
        """Calcula precisión por escenario"""
        print("\n" + "="*60)
        print("📊 PRECISIÓN POR ESCENARIO")
        print("="*60)
        
        # Filtrar solo registros con predicciones
        df_predictions = self.df[
            (self.df['objeto_verdad'].notna()) & 
            (self.df['objeto_predicho'].notna())
        ].copy()
        
        if df_predictions.empty:
            print("⚠️ No hay datos de predicción disponibles")
            return None
        
        # Calcular aciertos (exactos o similitud de texto)
        df_predictions['acierto_calc'] = df_predictions.apply(
            lambda row: self.calculate_text_similarity(
                str(row['objeto_verdad']), 
                str(row['objeto_predicho'])
            ), axis=1
        )
        
        # Agrupar por escenario
        precision_por_escenario = df_predictions.groupby('escenario').agg({
            'acierto_calc': ['sum', 'count', 'mean']
        })
        
        precision_por_escenario.columns = ['aciertos_totales', 'total_pruebas', 'similitud_promedio']
        precision_por_escenario['precision_pct'] = (
            precision_por_escenario['similitud_promedio'] * 100
        ).round(2)
        
        print(precision_por_escenario)
        
        # Precisión general
        precision_general = df_predictions['acierto_calc'].mean() * 100
        
        print(f"\n🎯 PRECISIÓN GENERAL (Similitud): {precision_general:.2f}%")
        
        return precision_por_escenario
    
    def calculate_text_similarity(self, ground_truth: str, predicted: str) -> float:
        """
        Calcula similitud entre dos textos usando distancia de Levenshtein normalizada
        Retorna un valor entre 0.0 (completamente diferente) y 1.0 (idéntico)
        """
        # Normalizar textos
        gt = str(ground_truth).lower().strip()
        pred = str(predicted).lower().strip()
        
        # Si son idénticos
        if gt == pred:
            return 1.0
        
        # Si alguno está vacío
        if not gt or not pred:
            return 0.0
        
        # Calcular distancia de Levenshtein
        distance = self.levenshtein_distance(gt, pred)
        
        # Normalizar por la longitud del texto más largo
        max_len = max(len(gt), len(pred))
        similarity = 1.0 - (distance / max_len)
        
        return max(0.0, similarity)  # Asegurar que no sea negativo
    
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calcula la distancia de Levenshtein entre dos strings"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Costo de inserción, eliminación o sustitución
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def analyze_latency_by_mode(self):
        """Analiza latencia por modo de ejecución"""
        print("\n" + "="*60)
        print("⏱️ LATENCIA POR MODO")
        print("="*60)
        
        df_latency = self.df[self.df['latencia_ms'].notna()].copy()
        
        if df_latency.empty:
            print("⚠️ No hay datos de latencia disponibles")
            return None
        
        latency_by_mode = df_latency.groupby('modo')['latencia_ms'].agg([
            'mean', 'median', 'std', 'min', 'max', 'count'
        ]).round(2)
        
        latency_by_mode.columns = [
            'Media (ms)', 'Mediana (ms)', 'Desv. Est.', 
            'Mín (ms)', 'Máx (ms)', 'N° Pruebas'
        ]
        
        print(latency_by_mode)
        
        return latency_by_mode
    
    def analyze_latency_by_component(self):
        """Analiza latencia por componente"""
        print("\n" + "="*60)
        print("⚙️ LATENCIA POR COMPONENTE")
        print("="*60)
        
        df_component = self.df[self.df['t_total_ms_numeric'].notna()].copy()
        
        if df_component.empty:
            print("⚠️ No hay datos de tiempo por componente")
            return None
        
        latency_by_component = df_component.groupby('tipo_evento')['t_total_ms_numeric'].agg([
            'mean', 'median', 'std', 'count'
        ]).round(2)
        
        latency_by_component.columns = [
            'Media (ms)', 'Mediana (ms)', 'Desv. Est.', 'N° Eventos'
        ]
        latency_by_component = latency_by_component.sort_values('Media (ms)', ascending=False)
        
        print(latency_by_component)
        
        return latency_by_component
    
    def analyze_errors(self):
        """Analiza tasa de errores por escenario"""
        print("\n" + "="*60)
        print("❌ ANÁLISIS DE ERRORES")
        print("="*60)
        
        # Identificar errores en notas
        df_errors = self.df.copy()
        df_errors['es_error'] = df_errors['notas'].fillna('').str.lower().str.contains(
            'error|crash|timeout|fail'
        ).astype(int)
        
        error_by_scenario = df_errors.groupby('escenario').agg({
            'es_error': 'sum',
            'id_prueba': 'count'
        })
        
        error_by_scenario.columns = ['Errores', 'Total_Pruebas']
        error_by_scenario['Tasa_Fallos_pct'] = (
            (error_by_scenario['Errores'] / error_by_scenario['Total_Pruebas']) * 100
        ).round(2)
        
        print(error_by_scenario)
        
        # Tasa general
        errores_total = error_by_scenario['Errores'].sum()
        pruebas_total = error_by_scenario['Total_Pruebas'].sum()
        tasa_general = (errores_total / pruebas_total * 100) if pruebas_total > 0 else 0
        
        print(f"\n⚠️ TASA DE FALLOS GENERAL: {tasa_general:.2f}%")
        
        return error_by_scenario
    
    def plot_all_metrics(self):
        """Genera todos los gráficos"""
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Precisión por escenario
        ax1 = plt.subplot(2, 3, 1)
        precision_data = self.calculate_precision()
        if precision_data is not None and not precision_data.empty:
            precision_data['precision_pct'].plot(
                kind='bar', ax=ax1, color='steelblue', alpha=0.7
            )
            ax1.set_title('Precisión por Escenario', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Escenario')
            ax1.set_ylabel('Precisión (%)')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=80, color='g', linestyle='--', alpha=0.5, label='Target 80%')
            ax1.legend()
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. Latencia por modo
        ax2 = plt.subplot(2, 3, 2)
        latency_mode = self.analyze_latency_by_mode()
        if latency_mode is not None and not latency_mode.empty:
            latency_mode['Media (ms)'].plot(
                kind='bar', ax=ax2, color='coral', alpha=0.7
            )
            ax2.set_title('Latencia Media por Modo', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Modo')
            ax2.set_ylabel('Latencia (ms)')
            ax2.grid(True, alpha=0.3)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Latencia por componente
        ax3 = plt.subplot(2, 3, 3)
        latency_comp = self.analyze_latency_by_component()
        if latency_comp is not None and not latency_comp.empty:
            latency_comp['Media (ms)'].plot(
                kind='barh', ax=ax3, color='mediumpurple', alpha=0.7
            )
            ax3.set_title('Latencia Media por Componente', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Latencia (ms)')
            ax3.set_ylabel('Componente')
            ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. Tasa de errores
        ax4 = plt.subplot(2, 3, 4)
        error_data = self.analyze_errors()
        if error_data is not None and not error_data.empty:
            error_data['Tasa_Fallos_pct'].plot(
                kind='bar', ax=ax4, color='indianred', alpha=0.7
            )
            ax4.set_title('Tasa de Fallos por Escenario', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Escenario')
            ax4.set_ylabel('Tasa de Fallos (%)')
            ax4.grid(True, alpha=0.3)
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 5. Distribución de latencias
        ax5 = plt.subplot(2, 3, 5)
        df_latency = self.df[self.df['latencia_ms'].notna()]
        if not df_latency.empty:
            for modo in df_latency['modo'].unique():
                data = df_latency[df_latency['modo'] == modo]['latencia_ms']
                ax5.hist(data, bins=20, alpha=0.5, label=modo)
            ax5.set_title('Distribución de Latencias', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Latencia (ms)')
            ax5.set_ylabel('Frecuencia')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Confianza de predicciones
        ax6 = plt.subplot(2, 3, 6)
        df_confidence = self.df[self.df['confianza'].notna()]
        if not df_confidence.empty:
            ax6.hist(df_confidence['confianza'], bins=20, 
                    color='teal', alpha=0.7, edgecolor='black')
            ax6.set_title('Distribución de Confianza', fontsize=12, fontweight='bold')
            ax6.set_xlabel('Confianza')
            ax6.set_ylabel('Frecuencia')
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('analisis_resultados.png', dpi=300, bbox_inches='tight')
        print("\n📊 Gráficos guardados en: analisis_resultados.png")
        plt.show()
    
    def generate_report(self):
        """Genera un reporte completo"""
        if not self.load_data():
            return
        
        print("\n" + "="*60)
        print("📋 REPORTE DE ANÁLISIS DE PRUEBAS")
        print("="*60)
        
        # Métricas generales
        self.calculate_precision()
        self.analyze_latency_by_mode()
        self.analyze_latency_by_component()
        self.analyze_errors()
        
        # Generar gráficos
        self.plot_all_metrics()
        
        print("\n✅ Análisis completado")


if __name__ == "__main__":
    analyzer = TestAnalyzer(csv_path="resultados_pruebas.csv")
    analyzer.generate_report()