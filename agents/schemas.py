"""구조화 출력 스키마 (Critic 등)."""

from pydantic import BaseModel, Field


class Critic1Result(BaseModel):
    """1차: 공정성·근거성 (및 수집 단계 완결성)."""

    fairness_ok: bool = Field(description="양사 균형·동일 비교축이면 True")
    grounding_ok: bool = Field(description="주요 주장에 출처가 있으면 True")
    feedback: str = Field(description="미통과 시 구체적 수정·재수집 지시")
    retry_target: str = Field(
        description='재수집 대상: "none"|"skon"|"catl"|"both"'
    )

    @property
    def pass_all(self) -> bool:
        return self.fairness_ok and self.grounding_ok


class Critic2Result(BaseModel):
    """2차: 논리성·일관성."""

    logic_ok: bool = Field(description="논리 비약이 없으면 True")
    consistency_ok: bool = Field(description="비교 기준 일관·모순 없으면 True")
    feedback: str = Field(description="실패 시 수정 지시, 통과 시 짧은 요약")

    @property
    def pass_all(self) -> bool:
        return self.logic_ok and self.consistency_ok


class Critic3Result(BaseModel):
    """3차: 보고서 초안 형식·내용 (목차 준수, SWOT/REFERENCE 등)."""

    format_ok: bool = Field(
        description="Markdown 목차·헤딩·표 형식, SUMMARY/REFERENCE 존재 등 형식 요건이면 True"
    )
    content_ok: bool = Field(
        description="T4 분석과 모순 없고 빈 섹션·환각적 추가가 없으면 True"
    )
    feedback: str = Field(description="미통과 시 Writer가 고칠 구체 항목. 통과 시 짧은 확인.")

    @property
    def pass_all(self) -> bool:
        return self.format_ok and self.content_ok
