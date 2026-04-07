"""구조화 출력 스키마 (Critic 등)."""

from pydantic import BaseModel, Field


class Critic1Result(BaseModel):
    """1차: 완결성·공정성·근거성 (수집 단계)."""

    completeness_ok: bool = Field(
        description="시장 맥락이 실질적으로 채워졌고, SK On·CATL 각각 핵심 축(기술/전략/시장 등) 데이터가 비어 있지 않으면 True"
    )
    fairness_ok: bool = Field(
        description="한쪽만 과도하게 미화/비하하지 않고, 동일한 비교 관점이 양사에 적용되면 True"
    )
    grounding_ok: bool = Field(
        description="정량·중대 주장에 출처(sources 등)가 붙어 있고, 근거 없는 단정이 없으면 True"
    )
    rationale_brief: str = Field(
        description="판정 근거를 2~4문장으로. 불통과 시 어떤 필드/문장이 문제인지 구체적으로."
    )
    feedback: str = Field(
        description="미통과 시: 번호 목록으로 재수집 지시. 통과 시: 한 줄 확인."
    )
    retry_target: str = Field(
        description='재수집 대상만: "none"|"skon"|"catl"|"both". 공정성만 문제면 어느 쪽 재수집이 유효한지 선택.'
    )

    @property
    def pass_all(self) -> bool:
        return self.completeness_ok and self.fairness_ok and self.grounding_ok


class Critic2Result(BaseModel):
    """2차: 논리·내부 일관성·수집 데이터 정합."""

    logic_ok: bool = Field(
        description="비약·순환논리·원인-결과 오류 없이, 주장이 앞선 근거에서 따라오면 True"
    )
    consistency_ok: bool = Field(
        description="같은 지표/사실에 대해 문단 간 서술이 모순되지 않으면 True"
    )
    data_alignment_ok: bool = Field(
        description="분석문의 사실·수치·서술이 제공된 수집 JSON/시장 맥락에 없는 내용을 단정하지 않으면 True"
    )
    rationale_brief: str = Field(
        description="2~4문장. 문제 구간(문장 인용 가능)을 짚어 설명."
    )
    feedback: str = Field(
        description="미통과 시: Analysis Agent가 고칠 항목을 번호 목록. 통과 시: 짧은 확인."
    )

    @property
    def pass_all(self) -> bool:
        return self.logic_ok and self.consistency_ok and self.data_alignment_ok


class Critic3Result(BaseModel):
    """3차: 형식·본문 정합·REFERENCE 무결성."""

    format_ok: bool = Field(
        description="SUMMARY, 다단 헤딩 목차, SWOT 표(또는 동등 구조), REFERENCE 섹션 등 형식 요건 충족이면 True"
    )
    content_ok: bool = Field(
        description="빈 장·한 줄 placeholder 없이, T4 분석·수집 요약과 명백히 모순되는 서술이 없으면 True"
    )
    reference_integrity_ok: bool = Field(
        description="REFERENCE·본문의 출처 표기가 수집 데이터에 존재하는 출처 범위를 벗어나지 않으면 True"
    )
    rationale_brief: str = Field(
        description="2~4문장. 형식/내용/출력 중 무엇을 어떻게 검토했는지."
    )
    feedback: str = Field(
        description="미통과 시: Writer가 수정할 항목을 번호 목록. 통과 시: 한 줄."
    )

    @property
    def pass_all(self) -> bool:
        return self.format_ok and self.content_ok and self.reference_integrity_ok
