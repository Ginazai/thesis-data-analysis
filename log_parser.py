"""
Parser de logs de pruebas de visión artificial a CSV
Procesa archivos .log/.txt y genera un CSV estructurado con métricas de pruebas
Soporta dos formatos:
  - MOBILE (.txt): [ISO8601] COMPONENT: event - details
  - ESP (.log): T+HH:MM:SS.mmm | COMPONENT | message
"""

import pandas as pd
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

class LogParser:
    def __init__(self, logs_directory: str = "./logs"):
        self.logs_dir = Path(logs_directory)
        self.id_counter = 1
        
    def parse_mobile_line(self, line: str) -> Optional[Dict]:
        """Parsea línea formato MOBILE: [timestamp] COMP: event - details O [timestamp] message"""
        # Intentar con componente y guión primero
        pattern = r'\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)\]\s+(\w+):\s+(.*?)\s+-\s+(.*)'
        match = re.match(pattern, line)
        
        if match:
            timestamp_str, component, event, details = match.groups()
            return {
                'timestamp': datetime.fromisoformat(timestamp_str),
                'component': component,
                'event': event.strip(),
                'details': details.strip(),
                'format': 'mobile'
            }
        
        # Si no hay guión, intentar sin él (con componente)
        pattern_no_dash = r'\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)\]\s+(\w+):\s+(.*)'
        match = re.match(pattern_no_dash, line)
        
        if match:
            timestamp_str, component, event_and_details = match.groups()
            return {
                'timestamp': datetime.fromisoformat(timestamp_str),
                'component': component,
                'event': event_and_details.strip(),
                'details': '',
                'format': 'mobile'
            }
        
        # Si no tiene componente, es un mensaje genérico (Resizing, Running inference, etc.)
        pattern_generic = r'\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)\]\s+(.*)'
        match = re.match(pattern_generic, line)
        
        if match:
            timestamp_str, message = match.groups()
            # Determinar componente basado en el mensaje
            component = 'GENERIC'
            if 'resizing' in message.lower():
                component = 'PREPROCESSING'
            elif 'running inference' in message.lower():
                component = 'INFERENCE'
            elif 'depth' in message.lower():
                component = 'DEPTH'
            
            return {
                'timestamp': datetime.fromisoformat(timestamp_str),
                'component': component,
                'event': message.strip(),
                'details': '',
                'format': 'mobile'
            }
        
        return None
    
    def parse_esp_line(self, line: str, base_time: datetime) -> Optional[Dict]:
        """Parsea línea formato ESP: T+HH:MM:SS.mmm | COMPONENT | message"""
        pattern = r'T\+(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s+\|\s+(\w+)\s+\|\s+(.*)'
        match = re.match(pattern, line)
        
        if match:
            hours, minutes, seconds, millis, component, message = match.groups()
            
            # Calcular timestamp relativo
            delta = timedelta(
                hours=int(hours),
                minutes=int(minutes),
                seconds=int(seconds),
                milliseconds=int(millis)
            )
            timestamp = base_time + delta
            
            return {
                'timestamp': timestamp,
                'component': component,
                'event': 'capture',
                'details': message.strip(),
                'format': 'esp'
            }
        return None
    
    def calculate_duration_ms(self, start_time: datetime, end_time: datetime) -> float:
        """Calcula duración en milisegundos entre dos timestamps"""
        delta = end_time - start_time
        return round(delta.total_seconds() * 1000, 2)
    
    def ms_to_hhmmss(self, milliseconds: float) -> str:
        """Convierte milisegundos a formato HH:MM:SS.mmm"""
        if milliseconds is None or pd.isna(milliseconds):
            return None
        
        try:
            milliseconds = float(milliseconds)
            total_seconds = milliseconds / 1000
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            ms = int((milliseconds % 1000))
            
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"
        except (ValueError, TypeError):
            return None
    
    def parse_file_name(self, filename: str) -> Dict:
        """Extrae información del nombre del archivo"""
        parts = filename.replace('.log', '').replace('.txt', '').split(' - ')
        
        modo = parts[0].strip() if parts else 'UNKNOWN'
        escenarios = ', '.join(parts[1:]) if len(parts) > 1 else ''
        
        return {
            'modo': modo,
            'escenarios': escenarios
        }
    
    def process_mobile_file(self, filepath: Path, parsed_lines: List[Dict]) -> List[Dict]:
        """Procesa archivo formato MOBILE (con timestamps completos)"""
        records = []
        file_info = self.parse_file_name(filepath.name)
        
        if not parsed_lines:
            return records
        
        # Variables para seguimiento de ciclos
        cycle_start_time = None
        cycle_start_idx = None
        
        for i, current in enumerate(parsed_lines):
            # Detectar inicio de ciclo
            is_capture_start = (
                current['component'] == 'ESP32' and 
                'capture' in current['event'] and 
                ('starting' in current['details'] or 'starting' in current['event'])
            )
            
            if is_capture_start:
                cycle_start_time = current['timestamp']
                cycle_start_idx = i
            
            # Calcular t_total_ms (tiempo desde evento anterior)
            t_total = None
            if i > 0:
                t_total = self.calculate_duration_ms(parsed_lines[i-1]['timestamp'], current['timestamp'])
            
            # Detectar si es el final del ciclo (TTS converted)
            is_cycle_end = (
                current['component'] == 'TTS' and 
                'converted output to speech' in current['event']
            )
            
            # Calcular latencia SOLO si es el final del ciclo Y hay un ciclo activo
            latencia = None
            if is_cycle_end and cycle_start_time is not None:
                latencia = self.calculate_duration_ms(cycle_start_time, current['timestamp'])
                # IMPORTANTE: Resetear el ciclo
                cycle_start_time = None
                cycle_start_idx = None
            
            # Buscar datos de SCENE
            objeto_predicho = None
            confianza = None
            
            if current['component'] == 'SCENE' and 'result' in current['event']:
                match_label = re.search(r'result\s+-\s+(\w+)', current['details'])
                if match_label:
                    label = match_label.group(1)
                    objeto_predicho = label if label.lower() != 'unknown' else 'unknown'
                
                match_conf = re.search(r'confidence\s+([\d.]+)', current['details'])
                if match_conf:
                    confianza = float(match_conf.group(1))
            
            # Capturar detalles completos en notas
            notas = ''
            if current.get('notas'):  # Notas ya capturadas de líneas multilinea
                notas = current['notas']
            elif current['details']:
                notas = current['details']
            elif current['event']:
                notas = current['event']
            
            # Crear registro
            record = {
                'id_prueba': self.id_counter,
                'timestamp': current['timestamp'].isoformat(),
                'modo': file_info['modo'],
                'tipo_evento': current['component'],
                'escenario': file_info['escenarios'],
                'distancia_m': None,
                'iluminacion': None,
                'objeto_verdad': None,
                'objeto_predicho': objeto_predicho,
                'confianza': confianza,
                't_total_ms': t_total,
                'latencia': latencia,
                'acierto': None,
                'notas': notas
            }
            
            records.append(record)
            self.id_counter += 1
        
        return records
    
    def process_esp_file(self, filepath: Path, parsed_lines: List[Dict]) -> List[Dict]:
        """Procesa archivo formato ESP (solo capturas de cámara)"""
        records = []
        file_info = self.parse_file_name(filepath.name)
        
        # Para ESP, cada línea es una captura completa
        for i, line in enumerate(parsed_lines):
            # Calcular t_total_ms solo si hay evento anterior EN ESTE ARCHIVO
            t_total = None
            if i > 0:
                t_total = self.calculate_duration_ms(parsed_lines[i-1]['timestamp'], line['timestamp'])
            
            record = {
                'id_prueba': self.id_counter,
                'timestamp': line['timestamp'].isoformat(),
                'modo': file_info['modo'],
                'tipo_evento': 'ESP32',
                'escenario': file_info['escenarios'],
                'distancia_m': None,
                'iluminacion': None,
                'objeto_verdad': None,
                'objeto_predicho': None,
                'confianza': None,
                't_total_ms': t_total,
                'latencia': None,
                'acierto': None,
                'notas': line['details']
            }
            
            records.append(record)
            self.id_counter += 1
        
        return records
    
    def process_log_file(self, filepath: Path) -> List[Dict]:
        """Procesa un archivo de log (detecta formato automáticamente)"""
        # Intentar diferentes encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        lines = None
        
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    lines = f.readlines()
                break
            except UnicodeDecodeError:
                continue
        
        if lines is None:
            print(f"    ⚠️ No se pudo leer con encodings estándar, usando UTF-8 con errores reemplazados")
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
        
        if not lines:
            return []
        
        # Detectar formato
        first_line = lines[0].strip()
        is_mobile = first_line.startswith('[') and re.match(r'\[\d{4}-\d{2}-\d{2}T', first_line)
        is_esp = first_line.startswith('T+')
        
        # Parsear líneas según formato
        parsed_lines = []
        base_time = datetime(2025, 1, 1, 0, 0, 0)
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
            
            if is_mobile:
                parsed = self.parse_mobile_line(line)
                
                # Si no se puede parsear, podría ser una continuación de la línea anterior
                if not parsed and parsed_lines:
                    # Agregar como continuación de la última línea parseada
                    if parsed_lines[-1]['format'] == 'mobile':
                        if parsed_lines[-1]['notas']:
                            parsed_lines[-1]['notas'] += ' ' + line
                        else:
                            parsed_lines[-1]['notas'] = line
                    i += 1
                    continue
                
            elif is_esp:
                parsed = self.parse_esp_line(line, base_time)
            else:
                i += 1
                continue
            
            if parsed:
                # Agregar campo 'notas' para almacenar continuaciones
                parsed['notas'] = ''
                parsed_lines.append(parsed)
            
            i += 1
        
        if not parsed_lines:
            return []
        
        # Procesar según formato
        if is_mobile:
            return self.process_mobile_file(filepath, parsed_lines)
        elif is_esp:
            return self.process_esp_file(filepath, parsed_lines)
        
        return []
    
    def parse_all_logs(self, output_csv: str = "resultados_pruebas.xlsx"):
        """Procesa todos los archivos de log en el directorio"""
        all_records = []
        
        log_files = list(self.logs_dir.glob("*.log")) + list(self.logs_dir.glob("*.txt"))
        
        if not log_files:
            print(f"⚠️ No se encontraron archivos de log en {self.logs_dir}")
            return None
        
        print(f"📁 Procesando {len(log_files)} archivo(s)...")
        
        for log_file in log_files:
            print(f"  ⚙️ Procesando: {log_file.name}")
            try:
                records = self.process_log_file(log_file)
                all_records.extend(records)
                print(f"    ✅ {len(records)} registros extraídos")
            except Exception as e:
                print(f"    ❌ Error: {str(e)}")
                import traceback
                traceback.print_exc()
        
        if not all_records:
            print("\n⚠️ No se generaron registros.")
            return None
        
        df = pd.DataFrame(all_records)
        
        # Ordenar por timestamp ascendente
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp_dt').reset_index(drop=True)
        
        # RECALCULAR t_total_ms después de ordenar
        # Guardar latencia original (no debe cambiar)
        latencia_original = df['latencia'].copy()
        
        # Calcular t_total_ms basado en el orden cronológico real
        df['t_total_ms'] = None
        for i in range(1, len(df)):
            prev_time = df.loc[i-1, 'timestamp_dt']
            curr_time = df.loc[i, 'timestamp_dt']
            duration_ms = (curr_time - prev_time).total_seconds() * 1000
            df.loc[i, 't_total_ms'] = round(duration_ms, 2)
        
        # Restaurar latencia (solo debe estar en TTS converted)
        df['latencia'] = latencia_original
        
        # Eliminar columna temporal
        df = df.drop('timestamp_dt', axis=1)
        
        # Reasignar id_prueba después de ordenar
        df['id_prueba'] = range(1, len(df) + 1)
        
        # Convertir latencia y t_total_ms a formato HH:MM:SS.mmm
        if 'latencia' in df.columns:
            df['latencia'] = df['latencia'].apply(lambda x: self.ms_to_hhmmss(x) if pd.notna(x) else None)
        if 't_total_ms' in df.columns:
            df['t_total_ms'] = df['t_total_ms'].apply(lambda x: self.ms_to_hhmmss(x) if pd.notna(x) else None)
        
        # Guardar con encoding UTF-8 con BOM para mejor compatibilidad
        df.to_excel(output_csv, index=False, encoding='utf-8-sig')
        print(f"\n✅ Archivo generado: {output_csv}")
        print(f"📊 Total de registros: {len(df)}")
        
        return df


if __name__ == "__main__":
    parser = LogParser(logs_directory="./logs")
    df = parser.parse_all_logs(output_csv="resultados_pruebas.xlsx")
    
    if df is not None and not df.empty:
        print("\n" + "="*60)
        print("📊 RESUMEN DE DATOS")
        print("="*60)
        
        print(f"\n📋 Eventos por componente:")
        print(df['tipo_evento'].value_counts())
        
        print(f"\n🔧 Modos de ejecución:")
        print(df['modo'].value_counts())
        
        # Contar registros con latencia
        latencia_count = df['latencia'].notna().sum()
        if latencia_count > 0:
            print(f"\n⏱️ Ciclos completos (con latencia): {latencia_count}")
        
        # Contar registros con t_total_ms
        t_total_count = df['t_total_ms'].notna().sum()
        if t_total_count > 0:
            print(f"\n⚙️ Eventos con tiempo medido: {t_total_count}")
        
        # Predicciones
        predictions = df[df['objeto_predicho'].notna()]
        if not predictions.empty:
            print(f"\n🎯 Objetos detectados:")
            print(predictions['objeto_predicho'].value_counts())
        
        print(f"\n📄 Muestra:")
        print(df.head(10).to_string())