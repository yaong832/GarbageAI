"""
클래스 매핑 및 새 클래스 관리 모듈
TACO 탐지 결과를 기존 분류 클래스로 매핑하거나 새 클래스를 추가합니다.
"""

import os
import json
from datetime import datetime

class ClassMapper:
    """TACO 클래스를 분류 클래스로 매핑하는 클래스"""
    
    def __init__(self, mapping_file='class_mappings.json', pending_file='pending_classes.json'):
        """
        Args:
            mapping_file: 클래스 매핑 정보를 저장할 파일
            pending_file: 승인 대기 중인 새 클래스를 저장할 파일
        """
        self.mapping_file = mapping_file
        self.pending_file = pending_file
        
        # 기본 클래스 목록
        self.base_classes = ['battery', 'biological', 'cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        
        # 기본 매핑 테이블 (TACO → 분류 클래스)
        self.default_mappings = {
            # 명확한 매핑
            'Can': 'metal',
            'Drink can': 'metal',
            'Food Can': 'metal',
            'Bottle': 'plastic',
            'Clear plastic bottle': 'plastic',
            'Other plastic bottle': 'plastic',
            'Glass bottle': 'glass',
            'Plastic bag + wrapper': 'plastic',
            'Garbage bag': 'plastic',
            'Single-use carrier bag': 'plastic',
            'Cup': 'plastic',
            'Disposable plastic cup': 'plastic',
            'Paper cup': 'paper',
            'Glass cup': 'glass',
            'Cardboard': 'cardboard',
            'Corrugated carton': 'cardboard',
            'Paper bag': 'paper',
            'Normal paper': 'paper',
            'Magazine paper': 'paper',
            'Battery': 'battery',
            'Food waste': 'biological',
            'Disposable food container': 'plastic',
            'Foam food container': 'plastic',
            
            # 불확실한 것들 → trash
            'Other': 'trash',
            'Unlabeled litter': 'trash',
            'Cigarette': 'trash',
            'Shoe': 'trash',
            'Rope & strings': 'trash',
            'Scrap metal': 'metal',  # 또는 trash
            'Broken glass': 'glass',  # 또는 trash
        }
        
        # 사용자 정의 매핑 로드
        self.custom_mappings = self._load_mappings()
        
        # 승인 대기 중인 클래스 로드
        self.pending_classes = self._load_pending()
    
    def _load_mappings(self):
        """사용자 정의 매핑 로드"""
        if os.path.exists(self.mapping_file):
            with open(self.mapping_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_mappings(self):
        """매핑 저장"""
        with open(self.mapping_file, 'w', encoding='utf-8') as f:
            json.dump(self.custom_mappings, f, ensure_ascii=False, indent=2)
    
    def _load_pending(self):
        """승인 대기 중인 클래스 로드"""
        if os.path.exists(self.pending_file):
            with open(self.pending_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def _save_pending(self):
        """승인 대기 클래스 저장"""
        with open(self.pending_file, 'w', encoding='utf-8') as f:
            json.dump(self.pending_classes, f, ensure_ascii=False, indent=2)
    
    def normalize_class_name(self, class_name):
        """클래스 이름을 정규화 (폴더명으로 사용 가능하게)"""
        # 소문자 변환, 공백을 언더스코어로
        normalized = class_name.lower().replace(' ', '_').replace('+', 'plus')
        # 특수문자 제거
        normalized = ''.join(c if c.isalnum() or c == '_' else '' for c in normalized)
        return normalized
    
    def map_class(self, taco_class, confidence=1.0):
        """
        TACO 클래스를 분류 클래스로 매핑
        
        Args:
            taco_class: TACO 탐지 모델이 반환한 클래스 이름
            confidence: 탐지 신뢰도 (0.0 ~ 1.0)
        
        Returns:
            tuple: (mapped_class, status, needs_review)
                - mapped_class: 매핑된 클래스 이름
                - status: 'mapped', 'new_candidate', 'unknown'
                - needs_review: 사용자 확인이 필요한지 여부
        """
        # 1. 사용자 정의 매핑 확인 (우선순위)
        if taco_class in self.custom_mappings:
            return self.custom_mappings[taco_class], 'mapped', False
        
        # 2. 기본 매핑 확인
        if taco_class in self.default_mappings:
            mapped = self.default_mappings[taco_class]
            # 신뢰도가 낮으면 검토 필요
            needs_review = confidence < 0.7
            return mapped, 'mapped', needs_review
        
        # 3. 정규화된 이름으로 기존 클래스와 비교
        normalized = self.normalize_class_name(taco_class)
        
        # 유사한 기존 클래스 찾기
        for base_class in self.base_classes:
            if base_class in normalized or normalized in base_class:
                needs_review = confidence < 0.8
                return base_class, 'mapped', needs_review
        
        # 4. 새 클래스 후보
        # 이미 승인 대기 중인지 확인
        for pending in self.pending_classes:
            if pending['taco_class'] == taco_class:
                return pending['normalized_name'], 'pending', True
        
        # 새 후보 추가
        new_candidate = {
            'taco_class': taco_class,
            'normalized_name': normalized,
            'confidence': confidence,
            'first_seen': datetime.now().isoformat(),
            'count': 1
        }
        self.pending_classes.append(new_candidate)
        self._save_pending()
        
        return normalized, 'new_candidate', True
    
    def add_custom_mapping(self, taco_class, mapped_class):
        """사용자 정의 매핑 추가"""
        self.custom_mappings[taco_class] = mapped_class
        self._save_mappings()
        
        # 승인 대기 목록에서 제거
        self.pending_classes = [p for p in self.pending_classes if p['taco_class'] != taco_class]
        self._save_pending()
    
    def approve_new_class(self, taco_class, approved_name=None):
        """
        새 클래스를 승인하여 추가
        
        Args:
            taco_class: TACO 클래스 이름
            approved_name: 승인된 클래스 이름 (None이면 정규화된 이름 사용)
        """
        # 승인 대기 목록에서 찾기
        pending = None
        for p in self.pending_classes:
            if p['taco_class'] == taco_class:
                pending = p
                break
        
        if not pending:
            return False
        
        # 클래스 이름 결정
        if approved_name:
            class_name = self.normalize_class_name(approved_name)
        else:
            class_name = pending['normalized_name']
        
        # 기존 클래스 목록에 추가
        if class_name not in self.base_classes:
            self.base_classes.append(class_name)
        
        # 매핑 추가
        self.add_custom_mapping(taco_class, class_name)
        
        return True
    
    def reject_new_class(self, taco_class, mapped_to_class='trash'):
        """새 클래스를 거부하고 기존 클래스로 매핑"""
        self.add_custom_mapping(taco_class, mapped_to_class)
        return True
    
    def get_pending_classes(self):
        """승인 대기 중인 클래스 목록 반환"""
        return self.pending_classes
    
    def get_all_classes(self):
        """모든 클래스 목록 반환 (기존 + 승인된 새 클래스)"""
        return self.base_classes.copy()
    
    def get_class_statistics(self, data_folder):
        """클래스별 이미지 개수 통계"""
        stats = {}
        for class_name in self.get_all_classes():
            class_folder = os.path.join(data_folder, class_name)
            if os.path.exists(class_folder):
                count = len([f for f in os.listdir(class_folder) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])
                stats[class_name] = count
            else:
                stats[class_name] = 0
        return stats

